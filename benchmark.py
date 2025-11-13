"""
NEEDLE Benchmark Runner

This script benchmarks OpenCV and MediaPipe eye detectors across scenarios and
produces the four artifacts per scenario:
 - opencv.png: OpenCV detection overlay
 - mediapipe.png: MediaPipe detection overlay
 - report.png: Comparison summary (bars/tables)
 - graph.png: Line charts of performance over time
 - metrics.csv / metrics.json: Raw and summary metrics

Scenarios supported out of the box:
 - light (bright lighting)
 - dim (low lighting)
 - glasses (user wears glasses)

Usage examples:
  python benchmark.py --scenario light --duration 20
  python benchmark.py --scenario dim --duration 20 --camera 0
  python benchmark.py --scenario glasses --duration 30 --no-show

If you have prerecorded videos, pass --video <path> to evaluate a file instead
of webcam. The outputs will be saved to tested/<scenario>/.
"""

import os
import cv2
import time
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import platform
from typing import Dict, List, Optional, Tuple

from opencv_detector import OpenCVEyeDetector
from mediapipe_detector import MediaPipeEyeDetector
from config import Config


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_detection_accuracy(liveness_results: List[Dict]) -> float:
    """
    Define detection accuracy as: fraction of frames where at least one eye produced
    a valid liveness result. We treat per-frame aggregation: 1 for any-eye result, else 0.
    """
    if not liveness_results:
        return 0.0
    return float(np.mean(np.array(liveness_results, dtype=float)))


def compute_stability_metrics(ears: List[float], pupil_centers: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Stability is about low variability and consistent tracking. We compute:
      - ear_std: standard deviation of EAR
      - pupil_jitter: mean Euclidean displacement between consecutive pupil centers
      - ear_iqr: interquartile range of EAR
    Lower is better for std/jitter/iqr. We also provide a normalized stability score
    where higher is better.
    """
    metrics = {
        'ear_std': float(np.nan),
        'ear_iqr': float(np.nan),
        'pupil_jitter': float(np.nan),
        'stability_score': float(np.nan)
    }

    valid_ears = [e for e in ears if e is not None and not math.isnan(e) and e > 0]
    if len(valid_ears) >= 5:
        arr = np.array(valid_ears, dtype=float)
        metrics['ear_std'] = float(np.std(arr))
        q75, q25 = np.percentile(arr, [75 ,25])
        metrics['ear_iqr'] = float(q75 - q25)

    # Pupil jitter
    disps = []
    prev = None
    for c in pupil_centers:
        if c is None:
            prev = None
            continue
        if prev is not None:
            disps.append(math.hypot(c[0]-prev[0], c[1]-prev[1]))
        prev = c
    if len(disps) > 0:
        metrics['pupil_jitter'] = float(np.mean(disps))

    # Heuristic stability score (0-1): invert-variance and invert-jitter with soft bounds
    ear_std = metrics['ear_std'] if not math.isnan(metrics['ear_std']) else 0.2
    jitter = metrics['pupil_jitter'] if not math.isnan(metrics['pupil_jitter']) else 3.0
    # Map typical ranges: ear_std ~ [0.01, 0.2], jitter ~ [0.5, 5.0]
    ear_term = 1.0 - min(max((ear_std - 0.01) / (0.2 - 0.01), 0.0), 1.0)
    jit_term = 1.0 - min(max((jitter - 0.5) / (5.0 - 0.5), 0.0), 1.0)
    metrics['stability_score'] = float(0.5*ear_term + 0.5*jit_term)

    return metrics


def draw_and_capture_sample(frame: np.ndarray, detector_name: str, detector, results: Dict, out_path: str):
    annotated = detector.draw_results(frame, results)
    cv2.imwrite(out_path, annotated)


def _apply_overlay_normalization(img: np.ndarray, gamma: float = 1.0, use_clahe: bool = False) -> np.ndarray:
    """Optionally brighten overlays without affecting metrics.
    - gamma > 1.0 brightens (gamma correction in linearized fashion)
    - use_clahe applies CLAHE to the luma channel
    """
    if img is None:
        return img
    out = img.copy()
    try:
        if use_clahe:
            ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            ycrcb[:, :, 0] = y
            out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        if gamma and abs(gamma - 1.0) > 1e-3:
            inv_gamma = 1.0 / max(gamma, 1e-6)
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            out = cv2.LUT(out, table)
    except Exception:
        # Fallback to original if any operation fails
        out = img
    return out


def _stamp_overlay_info(img: np.ndarray, scenario: str, detector_name: str) -> np.ndarray:
    """Add small text with scenario and mean luma to the overlay image for diagnostics."""
    if img is None:
        return img
    try:
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        mean_luma = float(np.mean(y))
        text = f"{scenario} | {detector_name} | meanY: {mean_luma:.1f}"
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception:
        pass
    return img


def run_on_stream(cap, detector_name: str, detector, duration: int, show: bool,
                  overlay_sample_every: int = 30,
                  warmup_sec: float = 2.5) -> Dict:
    start = time.time()
    frame_idx = 0

    # Warm-up: attempt to grab frames for a while to let auto-exposure/white-balance settle
    warmup_end = time.time() + max(0.0, float(warmup_sec))
    while time.time() < warmup_end:
        ret, frame_wu = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        if show:
            try:
                cv2.imshow(f'{detector_name} Warmup', frame_wu)
                cv2.waitKey(1)
            except Exception:
                pass

    per_frame = []  # rows dict
    detection_hits = []
    ears = []
    pupil_centers = []

    overlay_sample = None
    last_frame = None
    last_results = None

    while time.time() - start < duration:
        # Robust read with small retries
        ret, frame = cap.read()
        retry = 0
        while not ret and retry < 5:
            time.sleep(0.02)
            ret, frame = cap.read()
            retry += 1
        if not ret:
            # Could not read this iteration; continue loop instead of breaking to allow future frames
            continue

        t0 = time.time()
        results = detector.process_frame(frame)
        dt = time.time() - t0
        last_frame = frame
        last_results = results

        # Aggregate per-frame
        liveness = results.get('liveness_results', [])
        hit = 1.0 if len(liveness) > 0 else 0.0
        detection_hits.append(hit)

        # Average EAR across eyes for stability series
        if liveness:
            ear_vals = [lr.get('ear', np.nan) for lr in liveness]
            ear = float(np.nanmean(ear_vals)) if np.any(~np.isnan(ear_vals)) else np.nan
        else:
            ear = np.nan
        ears.append(ear)

        # Average pupil center (if available)
        centers = []
        for lr in liveness:
            pi = lr.get('pupil_info')
            if pi is not None:
                centers.append((float(pi[0]), float(pi[1])))
        pupil_centers.append(tuple(np.nanmean(centers, axis=0)) if len(centers) > 0 else None)

        perf = results.get('performance', {})
        per_frame.append({
            'frame': frame_idx,
            'proc_ms': dt*1000.0,
            'fps_reported': perf.get('fps', 0.0),
            'hit': hit,
            'ear': ear
        })

        # Periodic overlay capture
        if overlay_sample is None and frame_idx % overlay_sample_every == 0:
            overlay_sample = detector.draw_results(frame, results)

        if show:
            disp = detector.draw_results(frame, results)
            cv2.imshow(f'{detector_name} Preview', disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    # Compute summaries
    proc_ms = [r['proc_ms'] for r in per_frame]
    avg_proc_ms = float(np.mean(proc_ms)) if len(proc_ms) else float('nan')
    fps = 1000.0 / avg_proc_ms if avg_proc_ms and not math.isnan(avg_proc_ms) and avg_proc_ms > 0 else 0.0
    accuracy = compute_detection_accuracy(detection_hits)
    stability = compute_stability_metrics(ears, pupil_centers)

    # Ensure we have an overlay sample image even if none captured during loop
    if overlay_sample is None:
        if last_frame is not None and last_results is not None:
            overlay_sample = detector.draw_results(last_frame, last_results)
        else:
            # Create a placeholder image
            h, w = 480, 640
            overlay_sample = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(overlay_sample, f"{detector_name}: no frames captured",
                        (20, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return {
        'per_frame': per_frame,
        'overlay_sample': overlay_sample,
        'avg_proc_ms': avg_proc_ms,
        'fps': fps,
        'accuracy': accuracy,
        'stability': stability
    }


def _preferred_backends_for_platform() -> List[int]:
    sys_name = platform.system()
    backends = []
    # Note: getattr returns None if backend is missing; we'll filter those out later
    if sys_name == 'Darwin':  # macOS
        backends = [getattr(cv2, 'CAP_AVFOUNDATION', None)]
    elif sys_name == 'Windows':
        backends = [getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_MSMF', None)]
    else:  # Linux/Other
        backends = [getattr(cv2, 'CAP_V4L2', None)]
    # Fallback to default backend indicated by 0 (special path)
    backends.append(0)
    # Filter out Nones
    return [b for b in backends if b is not None]


def _try_open_camera(index: int, backends: List[int]):
    last_err = None
    for be in backends:
        try:
            cap = cv2.VideoCapture(index) if be == 0 else cv2.VideoCapture(index, be)
            if cap.isOpened():
                return cap, be
            try:
                cap.release()
            except Exception:
                pass
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return None, None


def list_available_cameras(max_index: int = 5) -> List[int]:
    """Probe camera indices from 0..max_index and return those that open successfully."""
    ok = []
    backends = _preferred_backends_for_platform()
    print(f"Probing cameras with backends: {backends}")
    for idx in range(0, max_index + 1):
        cap, be = _try_open_camera(idx, backends)
        if cap is not None and cap.isOpened():
            print(f"  - index {idx} ✓ (backend={be})")
            ok.append(idx)
            cap.release()
        else:
            print(f"  - index {idx} ✗")
    return ok


def open_input_source(camera_index: int, video_path: Optional[str], cam_opts: Optional[Dict] = None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        return cap

    # Try preferred backends for this platform
    backends = _preferred_backends_for_platform()
    cap, used_backend = _try_open_camera(camera_index, backends)
    if cap is None or not cap.isOpened():
        # Auto-probe alternative indices 0..5
        for alt_idx in range(0, 6):
            if alt_idx == camera_index:
                continue
            cap, used_backend = _try_open_camera(alt_idx, backends)
            if cap is not None and cap.isOpened():
                print(f"Info: requested camera index {camera_index} not available; using {alt_idx} (backend={used_backend})")
                camera_index = alt_idx
                break
    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Cannot access any camera (tried index {camera_index} with backends {backends}). "
            f"Use --list-cameras to probe indices or provide --video <path>."
        )

    # Apply config if available (set only after open)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.FPS)
    except Exception:
        pass

    # Reduce internal buffer to minimize stale frames (if supported)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print(f"Opened camera index {camera_index} (backend={used_backend})")

    # Optionally print camera properties for diagnostics
    if cam_opts and cam_opts.get('print_props'):
        props = {
            'FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
            'FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
            'FPS': cv2.CAP_PROP_FPS,
            'BRIGHTNESS': getattr(cv2, 'CAP_PROP_BRIGHTNESS', None),
            'CONTRAST': getattr(cv2, 'CAP_PROP_CONTRAST', None),
            'SATURATION': getattr(cv2, 'CAP_PROP_SATURATION', None),
            'HUE': getattr(cv2, 'CAP_PROP_HUE', None),
            'GAIN': getattr(cv2, 'CAP_PROP_GAIN', None),
            'EXPOSURE': getattr(cv2, 'CAP_PROP_EXPOSURE', None),
            'AUTO_EXPOSURE': getattr(cv2, 'CAP_PROP_AUTO_EXPOSURE', None),
            'WHITE_BALANCE_BLUE_U': getattr(cv2, 'CAP_PROP_WHITE_BALANCE_BLUE_U', None),
            'AUTO_WB': getattr(cv2, 'CAP_PROP_AUTO_WB', None),
        }
        print("Camera properties:")
        for name, pid in props.items():
            if pid is None:
                continue
            try:
                val = cap.get(pid)
                if val is not None and val != 0 and not math.isnan(val):
                    print(f" - {name}: {val}")
                else:
                    print(f" - {name}: {val}")
            except Exception:
                pass

    return cap


def plot_report(opencv_summary: Dict, mp_summary: Dict, out_path: str, scenario: str):
    labels = ['Accuracy', 'Stability', 'Speed (FPS)']

    def nz(v, default=0.0):
        try:
            return float(v) if v is not None and not math.isnan(float(v)) else float(default)
        except Exception:
            return float(default)

    acc = [nz(opencv_summary.get('accuracy')), nz(mp_summary.get('accuracy'))]
    stab = [nz(opencv_summary['stability'].get('stability_score')), nz(mp_summary['stability'].get('stability_score'))]
    spd = [nz(opencv_summary.get('fps')), nz(mp_summary.get('fps'))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    x = np.arange(len(labels))
    width = 0.35
    ax0 = ax[0]
    ax0.bar(x - width/2, [acc[0], stab[0], spd[0]], width, label='OpenCV', color='#1f77b4')
    ax0.bar(x + width/2, [acc[1], stab[1], spd[1]], width, label='MediaPipe', color='#ff7f0e')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ymax = max([1.0] + [v for v in acc + stab + spd if not math.isnan(v)])
    ax0.set_ylim(0, 1.2 * ymax)
    ax0.set_title(f'NEEDLE Comparison - {scenario}')
    ax0.legend()
    ax0.grid(True, axis='y', alpha=0.3)

    # Table of summaries with NA-friendly formatting
    def fmt_pct(x):
        return f"{x*100:.1f}%" if not math.isnan(x) else "NA"
    def fmt3(x):
        return f"{x:.3f}" if not math.isnan(x) else "NA"
    def fmt2(x):
        return f"{x:.2f}" if not math.isnan(x) else "NA"
    def fmt1(x):
        return f"{x:.1f}" if not math.isnan(x) else "NA"

    ear_std_ocv = nz(opencv_summary['stability'].get('ear_std'), default=float('nan'))
    jitter_ocv = nz(opencv_summary['stability'].get('pupil_jitter'), default=float('nan'))
    ear_std_mp = nz(mp_summary['stability'].get('ear_std'), default=float('nan'))
    jitter_mp = nz(mp_summary['stability'].get('pupil_jitter'), default=float('nan'))

    col_labels = ['Detector','Accuracy','EAR std↓','Pupil jitter↓','FPS','Proc ms↓']
    table_data = [
        ['OpenCV', fmt_pct(acc[0]), fmt3(ear_std_ocv), fmt2(jitter_ocv), fmt1(spd[0]), fmt1(nz(opencv_summary.get('avg_proc_ms')))],
        ['MediaPipe', fmt_pct(acc[1]), fmt3(ear_std_mp), fmt2(jitter_mp), fmt1(spd[1]), fmt1(nz(mp_summary.get('avg_proc_ms')))]
    ]
    ax1 = ax[1]
    ax1.axis('off')
    table = ax1.table(cellText=table_data, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeseries(opencv_pf: List[Dict], mp_pf: List[Dict], out_path: str, scenario: str):
    # Prepare series
    o_frames = [r['frame'] for r in opencv_pf]
    o_ear = [r['ear'] if r['ear'] is not None else np.nan for r in opencv_pf]
    o_hit = [r['hit'] for r in opencv_pf]

    m_frames = [r['frame'] for r in mp_pf]
    m_ear = [r['ear'] if r['ear'] is not None else np.nan for r in mp_pf]
    m_hit = [r['hit'] for r in mp_pf]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # EAR trends
    ax[0].plot(o_frames, o_ear, label='OpenCV EAR', color='#1f77b4', alpha=0.8)
    ax[0].plot(m_frames, m_ear, label='MediaPipe EAR', color='#ff7f0e', alpha=0.8)
    ax[0].set_ylabel('EAR')
    ax[0].set_title(f'EAR and Detection Hits over Time - {scenario}')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Detection hit (accuracy over time)
    ax[1].plot(o_frames, o_hit, label='OpenCV hit', color='#1f77b4')
    ax[1].plot(m_frames, m_hit, label='MediaPipe hit', color='#ff7f0e')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Detection (1/0)')
    ax[1].set_yticks([0, 1])
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_csv(per_frame: List[Dict], out_path: str):
    columns = ['frame', 'proc_ms', 'fps_reported', 'hit', 'ear']
    if per_frame and len(per_frame) > 0:
        df = pd.DataFrame(per_frame)
        # ensure all expected columns exist
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)
    df.to_csv(out_path, index=False)


def run_scenario(scenario: str, duration: int, camera_index: int, video_path: Optional[str], show: bool,
                 warmup_sec: float = 2.5,
                 normalize_overlays: bool = False,
                 gamma: float = 1.0,
                 use_clahe: bool = False):
    out_dir = os.path.join('tested', scenario)
    ensure_dir(out_dir)

    # Instantiate detectors fresh per scenario to reset internal stats
    ocv = OpenCVEyeDetector()
    mpd = MediaPipeEyeDetector()

    # Run OpenCV with a fresh capture
    print(f"Running OpenCV for scenario '{scenario}'...")
    cap1 = open_input_source(camera_index, video_path, cam_opts={'print_props': True})
    ocv_summary = run_on_stream(cap1, 'OpenCV', ocv, duration, show, warmup_sec=warmup_sec)
    cap1.release()

    # Run MediaPipe with a fresh capture (rewind video or reopen camera)
    print(f"Running MediaPipe for scenario '{scenario}'...")
    cap2 = open_input_source(camera_index, video_path, cam_opts={'print_props': True})
    mp_summary = run_on_stream(cap2, 'MediaPipe', mpd, duration, show, warmup_sec=warmup_sec)
    cap2.release()

    if show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Save overlay samples (optionally normalized and stamped)
    if ocv_summary['overlay_sample'] is not None:
        img = ocv_summary['overlay_sample']
        if normalize_overlays:
            img = _apply_overlay_normalization(img, gamma=gamma, use_clahe=use_clahe)
        img = _stamp_overlay_info(img, scenario, 'OpenCV')
        cv2.imwrite(os.path.join(out_dir, 'opencv.png'), img)
    if mp_summary['overlay_sample'] is not None:
        img = mp_summary['overlay_sample']
        if normalize_overlays:
            img = _apply_overlay_normalization(img, gamma=gamma, use_clahe=use_clahe)
        img = _stamp_overlay_info(img, scenario, 'MediaPipe')
        cv2.imwrite(os.path.join(out_dir, 'mediapipe.png'), img)

    # Save per-frame CSVs
    save_csv(ocv_summary['per_frame'], os.path.join(out_dir, 'opencv_metrics.csv'))
    save_csv(mp_summary['per_frame'], os.path.join(out_dir, 'mediapipe_metrics.csv'))

    # Save summary JSON
    summary = {
        'scenario': scenario,
        'opencv': {
            'accuracy': ocv_summary['accuracy'],
            'stability': ocv_summary['stability'],
            'fps': ocv_summary['fps'],
            'avg_proc_ms': ocv_summary['avg_proc_ms'],
            'total_frames': len(ocv_summary['per_frame'])
        },
        'mediapipe': {
            'accuracy': mp_summary['accuracy'],
            'stability': mp_summary['stability'],
            'fps': mp_summary['fps'],
            'avg_proc_ms': mp_summary['avg_proc_ms'],
            'total_frames': len(mp_summary['per_frame'])
        }
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot report and graph
    plot_report(summary['opencv'], summary['mediapipe'], os.path.join(out_dir, 'report.png'), scenario)
    plot_timeseries(ocv_summary['per_frame'], mp_summary['per_frame'], os.path.join(out_dir, 'graph.png'), scenario)

    print(f"Completed scenario '{scenario}'. Outputs saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='NEEDLE Benchmark Runner')
    p.add_argument('--scenario', type=str, default='light', choices=['light', 'dim', 'glasses'],
                   help='Scenario to evaluate (determines output directory)')
    p.add_argument('--duration', type=int, default=20, help='Duration in seconds per detector')
    p.add_argument('--camera', type=int, default=Config.CAMERA_INDEX, help='Camera index for webcam input')
    p.add_argument('--video', type=str, default=None, help='Optional video file to evaluate instead of webcam')
    p.add_argument('--show', dest='show', action='store_true', help='Show live preview windows')
    p.add_argument('--no-show', dest='show', action='store_false', help='Disable preview windows')
    p.add_argument('--list-cameras', dest='list_cameras', action='store_true', help='Probe and list available camera indices and exit')
    p.add_argument('--probe', dest='list_cameras', action='store_true', help='Alias for --list-cameras')
    # Exposure/brightness handling for overlays and camera warm-up
    p.add_argument('--warmup-sec', type=float, default=2.5, help='Warm-up seconds to let auto-exposure settle')
    p.add_argument('--normalize-overlays', dest='normalize_overlays', action='store_true', help='Apply gamma/CLAHE to saved overlays (visual only)')
    p.add_argument('--no-normalize-overlays', dest='normalize_overlays', action='store_false', help='Do not normalize overlays')
    p.add_argument('--gamma', type=float, default=1.4, help='Gamma value (>1 brightens) used when normalizing overlays')
    p.add_argument('--clahe', dest='use_clahe', action='store_true', help='Use CLAHE on overlays when normalizing')
    p.add_argument('--no-clahe', dest='use_clahe', action='store_false', help='Disable CLAHE on overlays')
    p.set_defaults(show=False, list_cameras=False, normalize_overlays=False, use_clahe=False)
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_cameras and args.video is None:
        # Only probe cameras when not using a video file
        list_available_cameras(max_index=5)
        return
    try:
        run_scenario(
            args.scenario,
            args.duration,
            args.camera,
            args.video,
            args.show,
            warmup_sec=args.warmup_sec,
            normalize_overlays=args.normalize_overlays,
            gamma=args.gamma,
            use_clahe=args.use_clahe,
        )
    except RuntimeError as e:
        # Friendly error without Python traceback
        print(f"Error: {e}")
        print("Hints:"
              "\n - Try a different camera index (e.g., --camera 1 or --camera 2)."
              "\n - Run --list-cameras to see which indices are available."
              "\n - Or supply a video file with --video /path/to/file.mp4")
        sys.exit(2)


if __name__ == '__main__':
    main()
