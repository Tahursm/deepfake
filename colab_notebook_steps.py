"""
Google Colab Notebook - Step by Step Instructions
Copy each section into separate cells in Google Colab
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
# Install all required packages
!pip install -q torch torchvision torchaudio
!pip install -q opencv-python-headless
!pip install -q librosa
!pip install -q scikit-learn
!pip install -q matplotlib
!pip install -q tqdm
!pip install -q timm
!pip install -q transformers
!pip install -q mediapipe
!pip install -q face-alignment
!pip install -q pyyaml

print("✅ All dependencies installed!")
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
    print("📁 Project structure:")
    !ls -la
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

