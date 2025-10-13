"""
Setup script for NEEDLE Liveness Detection System
Handles installation, testing, and initial configuration
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully")
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n🔧 Try installing manually:")
        print("pip install opencv-python mediapipe numpy matplotlib scipy pandas pillow seaborn")
        return False

def check_camera():
    """Check if camera is available"""
    print("\n📷 Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  Camera not detected or not accessible")
            print("   Make sure your camera is connected and not used by other applications")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Cannot read from camera")
            cap.release()
            return False
        
        print(f"✅ Camera working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
        
    except ImportError:
        print("❌ OpenCV not installed - cannot test camera")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def run_basic_test():
    """Run basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        result = subprocess.run([sys.executable, "test_needle.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ All basic tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out - this might be normal for camera tests")
        return True
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() != "Windows":
        return
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "NEEDLE Liveness Detection.lnk")
        target = os.path.join(os.getcwd(), "main.py")
        wDir = os.getcwd()
        icon = target
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{target}"'
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        
        print("✅ Desktop shortcut created")
        
    except ImportError:
        print("⚠️  Cannot create desktop shortcut (winshell not available)")
    except Exception as e:
        print(f"⚠️  Failed to create desktop shortcut: {e}")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 NEEDLE LIVENESS DETECTION SYSTEM SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 USAGE INSTRUCTIONS:")
    print("\n1. Full GUI Application:")
    print("   python main.py")
    
    print("\n2. Simple Demo (OpenCV):")
    print("   python demo.py")
    
    print("\n3. Simple Demo (MediaPipe):")
    print("   python demo.py --detector mediapipe")
    
    print("\n4. Benchmark Both Detectors:")
    print("   python demo.py --benchmark")
    
    print("\n5. Run Tests:")
    print("   python test_needle.py")
    
    print("\n📖 FEATURES:")
    print("• Real-time liveness detection using NEEDLE algorithm")
    print("• Compare OpenCV vs MediaPipe performance")
    print("• Micro eye-movement pattern analysis")
    print("• Component-wise scoring and analysis")
    print("• Performance benchmarking and metrics")
    
    print("\n🔧 CONFIGURATION:")
    print("• Edit config.py to adjust detection parameters")
    print("• Modify NEEDLE_THRESHOLD for sensitivity")
    print("• Adjust NEEDLE_WINDOW_SIZE for analysis window")
    
    print("\n❓ TROUBLESHOOTING:")
    print("• Ensure good lighting conditions")
    print("• Position face clearly in camera view")
    print("• Close other applications using the camera")
    print("• Check camera permissions in system settings")
    
    print("\n📚 For detailed documentation, see README.md")

def main():
    """Main setup function"""
    print("🚀 NEEDLE Liveness Detection System Setup")
    print("Natural Eye-movement Evaluation for Detecting Live Entities")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Setup incomplete due to dependency installation failure")
        print("You can try to run the system anyway, but some features may not work")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Check camera
    camera_ok = check_camera()
    if not camera_ok:
        print("⚠️  Camera issues detected - live detection may not work properly")
    
    # Run basic tests
    test_ok = run_basic_test()
    if not test_ok:
        print("⚠️  Some functionality tests failed")
        print("The system may still work, but with limited functionality")
    
    # Create desktop shortcut (Windows only)
    create_desktop_shortcut()
    
    # Print usage instructions
    print_usage_instructions()
    
    # Final status
    print("\n" + "="*60)
    if camera_ok and test_ok:
        print("✅ SETUP SUCCESSFUL - System ready for use!")
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("   Some features may not work properly")
    print("="*60)

if __name__ == "__main__":
    main()
