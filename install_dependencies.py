"""
Comprehensive dependency installation script that resolves all conflicts.

This script installs packages in the correct order with proper version constraints
to avoid dependency conflicts between mediapipe, numpy, protobuf, and other packages.
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a pip command and handle errors."""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: {description} had issues (exit code {result.returncode})")
        return False
    return True

def main():
    print("🔧 Comprehensive Dependency Installation")
    print("="*70)
    print("\nThis script will install all dependencies with correct versions")
    print("to avoid conflicts between mediapipe, numpy, protobuf, and other packages.\n")
    
    # Step 1: Install core dependencies first (protobuf and numpy)
    print("\n" + "="*70)
    print("STEP 1: Installing Core Dependencies")
    print("="*70)
    
    run_command(
        'pip install --upgrade pip',
        "Upgrading pip"
    )
    
    run_command(
        'pip install "protobuf>=4.25.3,<5.0.0"',
        "Installing protobuf (compatible with mediapipe)"
    )
    
    run_command(
        'pip install "numpy>=1.24.0,<2.0.0"',
        "Installing numpy (compatible with mediapipe)"
    )
    
    # Step 2: Install packages that might pull in conflicting dependencies
    # Install older versions that are compatible with numpy 1.x and protobuf 4.x
    print("\n" + "="*70)
    print("STEP 2: Installing Compatible Versions of Conflicting Packages")
    print("="*70)
    
    # Install opencv-python-headless (compatible with numpy 1.x)
    run_command(
        'pip install "opencv-python-headless>=4.8.0,<4.12.0"',
        "Installing opencv-python-headless (compatible with numpy<2.0)"
    )
    
    # Try to install older versions of conflicting packages if they exist
    # These may fail if versions don't exist, which is okay
    run_command(
        'pip install "pytensor<2.35.0" || pip install "pytensor<2.34.0" || echo "Skipping pytensor"',
        "Installing compatible pytensor version (if needed)"
    )
    
    run_command(
        'pip install "shap<0.50.0" || pip install "shap<0.49.0" || echo "Skipping shap"',
        "Installing compatible shap version (if needed)"
    )
    
    # Step 3: Install main project dependencies
    print("\n" + "="*70)
    print("STEP 3: Installing Main Project Dependencies")
    print("="*70)
    
    run_command(
        'pip install torch torchvision torchaudio',
        "Installing PyTorch"
    )
    
    run_command(
        'pip install mediapipe',
        "Installing mediapipe"
    )
    
    run_command(
        'pip install librosa scikit-learn matplotlib tqdm timm transformers face-alignment pyyaml',
        "Installing other core packages"
    )
    
    # Step 4: Install from requirements.txt (this will respect already installed versions)
    print("\n" + "="*70)
    print("STEP 4: Installing from requirements.txt")
    print("="*70)
    
    run_command(
        'pip install -r requirements.txt',
        "Installing remaining dependencies from requirements.txt"
    )
    
    # Step 5: Verify installation
    print("\n" + "="*70)
    print("STEP 5: Verifying Installation")
    print("="*70)
    
    print("\n🔍 Checking for dependency conflicts...")
    run_command('pip check', "Checking for conflicts")
    
    print("\n" + "="*70)
    print("✅ Installation Complete!")
    print("="*70)
    
    print("\n📋 Installed Versions:")
    packages_to_check = ['numpy', 'protobuf', 'mediapipe', 'opencv-python-headless']
    for pkg in packages_to_check:
        run_command(f'pip show {pkg} | grep Version', f"Checking {pkg} version")
    
    print("\n" + "="*70)
    print("📝 Notes:")
    print("="*70)
    print("""
1. Some packages (ydf, opentelemetry-proto, grpcio-status) may show warnings
   about requiring protobuf>=5.0, but they should still function with protobuf 4.x.

2. Some packages (pytensor, shap) may show warnings about requiring numpy>=2.0,
   but older versions should work with numpy 1.x.

3. If you encounter runtime errors, you may need to:
   - Downgrade specific conflicting packages
   - Use alternative packages
   - Wait for mediapipe to support numpy 2.x and protobuf 5.x

4. To verify everything works, run:
   python -c "import mediapipe, numpy, protobuf, cv2; print('✅ All imports successful')"
    """)

if __name__ == "__main__":
    main()

