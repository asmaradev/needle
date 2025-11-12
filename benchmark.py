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


def run_on_stream(cap, detector_name: str, detector, duration: int, show: bool,
                  overlay_sample_every: int = 30) -> Dict:
    start = time.time()
    frame_idx = 0

    per_frame = []  # rows dict
    detection_hits = []
    ears = []
    pupil_centers = []

    overlay_sample = None
    sample_path = None

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = detector.process_frame(frame)
        dt = time.time() - t0

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

    return {
        'per_frame': per_frame,
        'overlay_sample': overlay_sample,
        'avg_proc_ms': avg_proc_ms,
        'fps': fps,
        'accuracy': accuracy,
        'stability': stability
    }


def open_input_source(camera_index: int, video_path: Optional[str]):
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        return cap
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access camera index {camera_index}")
    # Apply config if available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS)
    return cap


def plot_report(opencv_summary: Dict, mp_summary: Dict, out_path: str, scenario: str):
    labels = ['Accuracy', 'Stability', 'Speed (FPS)']
    acc = [opencv_summary['accuracy'], mp_summary['accuracy']]
    stab = [opencv_summary['stability']['stability_score'], mp_summary['stability']['stability_score']]
    spd = [opencv_summary['fps'], mp_summary['fps']]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    x = np.arange(len(labels))
    width = 0.35
    ax0 = ax[0]
    ax0.bar(x - width/2, [acc[0], stab[0], spd[0]], width, label='OpenCV', color='#1f77b4')
    ax0.bar(x + width/2, [acc[1], stab[1], spd[1]], width, label='MediaPipe', color='#ff7f0e')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.2*max(max(acc), max(stab), max(spd), 1))
    ax0.set_title(f'NEEDLE Comparison - {scenario}')
    ax0.legend()
    ax0.grid(True, axis='y', alpha=0.3)

    # Table of summaries
    col_labels = ['Detector','Accuracy','EAR std↓','Pupil jitter↓','FPS','Proc ms↓']
    table_data = [
        ['OpenCV', f"{opencv_summary['accuracy']*100:.1f}%", f"{opencv_summary['stability']['ear_std']:.3f}",
         f"{opencv_summary['stability']['pupil_jitter']:.2f}", f"{opencv_summary['fps']:.1f}", f"{opencv_summary['avg_proc_ms']:.1f}"],
        ['MediaPipe', f"{mp_summary['accuracy']*100:.1f}%", f"{mp_summary['stability']['ear_std']:.3f}",
         f"{mp_summary['stability']['pupil_jitter']:.2f}", f"{mp_summary['fps']:.1f}", f"{mp_summary['avg_proc_ms']:.1f}"]
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
    df = pd.DataFrame(per_frame)
    df.to_csv(out_path, index=False)


def run_scenario(scenario: str, duration: int, camera_index: int, video_path: Optional[str], show: bool):
    out_dir = os.path.join('tested', scenario)
    ensure_dir(out_dir)

    cap = open_input_source(camera_index, video_path)

    # Instantiate detectors fresh per scenario to reset internal stats
    ocv = OpenCVEyeDetector()
    mpd = MediaPipeEyeDetector()

    print(f"Running OpenCV for scenario '{scenario}'...")
    ocv_summary = run_on_stream(cap, 'OpenCV', ocv, duration, show)

    # Rewind or reopen for MediaPipe run
    if video_path:
        cap.release()
        cap = open_input_source(camera_index, video_path)
    else:
        # For webcam, run back-to-back; conditions may drift slightly
        pass

    print(f"Running MediaPipe for scenario '{scenario}'...")
    mp_summary = run_on_stream(cap, 'MediaPipe', mpd, duration, show)

    cap.release()
    if show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Save overlay samples
    if ocv_summary['overlay_sample'] is not None:
        cv2.imwrite(os.path.join(out_dir, 'opencv.png'), ocv_summary['overlay_sample'])
    if mp_summary['overlay_sample'] is not None:
        cv2.imwrite(os.path.join(out_dir, 'mediapipe.png'), mp_summary['overlay_sample'])

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
    p.set_defaults(show=False)
    return p.parse_args()


def main():
    args = parse_args()
    run_scenario(args.scenario, args.duration, args.camera, args.video, args.show)


if __name__ == '__main__':
    main()
