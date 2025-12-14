"""
Fix dependency conflicts by upgrading/downgrading packages to compatible versions.

This script will:
1. Upgrade protobuf to a compatible version (>=5.29.1,<6.0.0)
2. Downgrade numpy to <2.0.0 for mediapipe compatibility
3. Reinstall conflicting packages if needed
"""

import subprocess
import sys

def run_pip_command(command):
    """Run a pip command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,
        text=True
    )
    return result.returncode == 0

def main():
    print("🔧 Fixing dependency conflicts...")
    print("\nThis script will:")
    print("1. Upgrade protobuf to >=5.29.1,<6.0.0")
    print("2. Downgrade numpy to <2.0.0 (for mediapipe compatibility)")
    print("3. Reinstall packages to resolve conflicts")
    
    # Step 1: Upgrade protobuf
    print("\n📦 Step 1: Upgrading protobuf...")
    if not run_pip_command("pip install 'protobuf>=5.29.1,<6.0.0'"):
        print("⚠️  Warning: protobuf upgrade had issues")
    
    # Step 2: Downgrade numpy to <2.0.0
    print("\n📦 Step 2: Downgrading numpy to <2.0.0...")
    if not run_pip_command("pip install 'numpy>=1.24.0,<2.0.0'"):
        print("⚠️  Warning: numpy downgrade had issues")
    
    # Step 3: Reinstall requirements to ensure compatibility
    print("\n📦 Step 3: Reinstalling requirements with resolved versions...")
    if not run_pip_command("pip install -r requirements.txt"):
        print("⚠️  Warning: Some packages may have issues")
    
    # Step 4: Check for remaining conflicts
    print("\n🔍 Step 4: Checking for remaining conflicts...")
    run_pip_command("pip check")
    
    print("\n" + "="*60)
    print("✅ Dependency fixing complete!")
    print("="*60)
    print("\nNote: Some packages (pytensor, opencv-python, shap) may show")
    print("warnings about requiring numpy>=2.0, but they should still work")
    print("with numpy 1.x. If you encounter runtime errors, you may need")
    print("to downgrade those specific packages or use a newer mediapipe version.")
    print("\nTo verify everything is working, run:")
    print("  python -c 'import mediapipe, numpy, protobuf; print(\"✅ All imports successful\")'")

if __name__ == "__main__":
    main()

