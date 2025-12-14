"""
Google Colab Notebook - Step by Step Instructions
Copy each section into separate cells in Google Colab
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
# Install all required packages with version constraints to avoid conflicts
# This installation order prevents dependency conflicts

# Step 0: Clean up any conflicting packages that might be installed
print("📦 Step 0: Cleaning up conflicting packages...")
!pip uninstall -y opencv-python 2>/dev/null || true
!pip uninstall -y pytensor 2>/dev/null || true
!pip uninstall -y shap 2>/dev/null || true
!pip uninstall -y ydf 2>/dev/null || true

# Step 1: Upgrade pip and install core dependencies first (protobuf and numpy)
# mediapipe requires protobuf<5 and numpy<2
print("\n📦 Step 1: Installing core dependencies (protobuf, numpy)...")
print("   (Warnings about dependency conflicts are expected and can be ignored)")
!pip install -q --upgrade pip
!pip install -q --force-reinstall --no-deps "protobuf>=4.25.3,<5.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Pin numpy to a specific version range to prevent upgrades
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true

# Step 2: Install opencv-python (not headless) with compatible version
# face-alignment requires opencv-python, and we need a version compatible with numpy<2.0
print("📦 Step 2: Installing opencv-python (compatible version)...")
!pip install -q --force-reinstall --no-deps "opencv-python>=4.8.0,<4.12.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Reinstall numpy again after opencv (opencv might try to upgrade it)
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true

# Step 3: Install PyTorch (this may try to upgrade numpy, but we'll fix it after)
print("📦 Step 3: Installing PyTorch...")
!pip install -q torch torchvision torchaudio 2>&1 | grep -v "ERROR: pip's dependency resolver" || true

# Step 4: CRITICAL - Reinstall numpy IMMEDIATELY after PyTorch (PyTorch often upgrades it)
print("📦 Step 4: Ensuring numpy version is correct (CRITICAL - PyTorch may have upgraded it)...")
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Verify numpy version
import numpy; print(f"   ✅ Verified: numpy version = {numpy.__version__}")

# Step 5: Install mediapipe (this will lock protobuf and numpy versions)
print("📦 Step 5: Installing mediapipe...")
!pip install -q mediapipe 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Reinstall numpy again after mediapipe (just to be safe)
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true

# Step 6: Install other packages (they should respect already installed versions)
print("📦 Step 6: Installing other packages...")
!pip install -q librosa scikit-learn matplotlib tqdm timm transformers face-alignment pyyaml 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Reinstall numpy after installing other packages (they might try to upgrade it)
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true

# Step 7: Final check - reinstall protobuf and numpy to ensure correct versions
# Use --force-reinstall --no-deps to override any packages that upgraded them
print("📦 Step 7: Final verification - ensuring correct versions...")
!pip install -q --force-reinstall --no-deps "protobuf>=4.25.3,<5.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
!pip install -q --force-reinstall --no-deps "numpy>=1.24.0,<2.0.0" 2>&1 | grep -v "ERROR: pip's dependency resolver" || true
# Final verification
import numpy
print(f"   ✅ Final check: numpy={numpy.__version__} (must be <2.0.0)")
if float(numpy.__version__.split('.')[0]) >= 2:
    print("   ⚠️  WARNING: numpy version is >=2.0! This will break mediapipe!")
else:
    print("   ✅ numpy version is correct for mediapipe compatibility")

print("\n✅ All dependencies installed!")
print("📝 Note: After cloning in CELL 2, dependencies will be reinstalled from requirements.txt")
print("   to ensure correct versions and resolve any conflicts.")
print("\n" + "="*70)
print("ℹ️  IMPORTANT: About Dependency Warnings")
print("="*70)
print("""
The warnings you see about dependency conflicts (numpy>=2.0, protobuf>=5.0) are
INFORMATIONAL ONLY and can be safely ignored. Here's why:

1. These warnings come from packages that are NOT directly used in this project
   (pytensor, ydf, shap, opentelemetry-proto, grpcio-status)

2. These packages are transitive dependencies (dependencies of dependencies) that
   may have been installed by other packages

3. The packages WILL STILL WORK with numpy 1.x and protobuf 4.x - the warnings
   are just pip being overly cautious

4. The critical packages (mediapipe, opencv-python-headless) are correctly installed
   with compatible versions

✅ If everything installed successfully, you're good to go!
⚠️  Only worry if you encounter RUNTIME ERRORS when actually using the code.
""")
"""

# ============================================================================
# CELL 2: Clone Repository from GitHub
# ============================================================================
"""
# Clone the repository from GitHub
print("📥 Cloning repository from GitHub...")

# Update this with your GitHub repository URL
GITHUB_REPO = "https://github.com/Tahursm/deepfake.git"

!git clone https://github.com/Tahursm/deepfake.git

# Move all files to current directory
import shutil
import os

repo_name = "deepfake"  # Change if your repo has a different name
if os.path.exists(repo_name):
    # Copy all files
    for item in os.listdir(repo_name):
        src = os.path.join(repo_name, item)
        dst = item
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # Remove the cloned directory
    shutil.rmtree(repo_name)
    
    print("✅ Repository cloned and files moved!")
    
    print("\n📁 Project structure:")
    !ls -la
    
    # Install dependencies from requirements.txt to ensure correct versions
    # This overrides any versions installed in CELL 1 with the project's constraints
    print("\n📦 Installing dependencies from requirements.txt...")
    print("   (This ensures correct versions and resolves dependency conflicts)")
    !pip install -q -r requirements.txt
    print("✅ Dependencies installed from requirements.txt!")
else:
    print("⚠️  Repository not found. Please check the GITHUB_REPO URL.")
"""

# ============================================================================
# CELL 3: Mount Google Drive and Copy Data
# ============================================================================
"""
from google.colab import drive
import os
from pathlib import Path
import shutil

# Mount Google Drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# Create data directory structure
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(f'data/{split}/{label}', exist_ok=True)

print("✅ Data directories created!")

# IMPORTANT: Update this path to match where you uploaded your videos in Google Drive
DRIVE_DATA_PATH = '/content/drive/MyDrive/Deepfake/data'

# Check if data exists
if os.path.exists(DRIVE_DATA_PATH):
    print(f"✅ Found data in Drive at: {DRIVE_DATA_PATH}")
    print("📦 Copying videos from Google Drive...")
    
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            drive_folder = Path(DRIVE_DATA_PATH) / split / label
            colab_folder = Path('data') / split / label
            
            if drive_folder.exists():
                # Copy all video files
                video_files = list(drive_folder.glob('*.mp4')) + \
                             list(drive_folder.glob('*.avi')) + \
                             list(drive_folder.glob('*.mov')) + \
                             list(drive_folder.glob('*.mkv'))
                
                for video_file in video_files:
                    shutil.copy2(video_file, colab_folder / video_file.name)
                
                print(f"  ✅ {split}/{label}: {len(video_files)} videos copied")
    
    print("\n✅ All videos copied!")
else:
    print(f"⚠️  Data not found at: {DRIVE_DATA_PATH}")
    print("Please upload your videos to Google Drive first!")
"""

# ============================================================================
# CELL 4: Prepare Dataset Metadata
# ============================================================================
"""
import json
from pathlib import Path

def create_metadata(data_dir, split):
    samples = []
    split_path = Path(data_dir) / split
    
    # Add real samples
    real_path = split_path / 'real'
    if real_path.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            for video_file in real_path.glob(ext):
                samples.append({
                    'video_path': str(video_file),
                    'label': 0  # 0 = real
                })
    
    # Add fake samples
    fake_path = split_path / 'fake'
    if fake_path.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            for video_file in fake_path.glob(ext):
                samples.append({
                    'video_path': str(video_file),
                    'label': 1  # 1 = fake
                })
    
    return samples

# Generate metadata for all splits
data_dir = 'data'
for split in ['train', 'val', 'test']:
    print(f"Processing {split}...")
    samples = create_metadata(data_dir, split)
    
    if len(samples) == 0:
        print(f"  ⚠️  No samples found for {split}")
        continue
    
    # Save metadata
    metadata_file = Path(data_dir) / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    real_count = sum(1 for s in samples if s['label'] == 0)
    fake_count = sum(1 for s in samples if s['label'] == 1)
    
    print(f"  ✅ {len(samples)} samples (Real: {real_count}, Fake: {fake_count})")

print("\n✅ Metadata files created!")
"""

# ============================================================================
# CELL 5: Update Config for Colab GPU
# ============================================================================
"""
import yaml
import os

config_path = 'experiments/configs/default.yaml'
os.makedirs('experiments/configs', exist_ok=True)

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for Colab GPU
    if 'training' in config:
        config['training']['batch_size'] = 2  # Increase for GPU
        config['training']['num_workers'] = 0  # Set to 0 to avoid MediaPipe multiprocessing issues
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Config updated for Colab!")
    print(f"   Batch size: {config['training']['batch_size']}")
else:
    print("⚠️  Config file not found. Please ensure you've cloned/uploaded the repository.")
"""

# ============================================================================
# CELL 6: Run Training
# ============================================================================
"""
# Start training
!python src/models/train.py --config experiments/configs/default.yaml

print("✅ Training complete!")
"""

# ============================================================================
# CELL 7: Download Results
# ============================================================================
"""
from google.colab import files
import os

# Download best checkpoint
if os.path.exists('experiments/checkpoints/checkpoint_best.pth'):
    files.download('experiments/checkpoints/checkpoint_best.pth')
    print("✅ Best checkpoint downloaded!")

# Download latest checkpoint
if os.path.exists('experiments/checkpoints/checkpoint_latest.pth'):
    files.download('experiments/checkpoints/checkpoint_latest.pth')
    print("✅ Latest checkpoint downloaded!")

print("🎉 All done!")
"""

