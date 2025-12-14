"""
Google Colab Setup Script for Deepfake Detection Project
Run this in Google Colab to set up and train your model
"""

# ============================================================================
# STEP 6: SET UP DATA FROM GOOGLE DRIVE
# ============================================================================

print("=" * 60)
print("STEP 6: Setting up data from Google Drive")
print("=" * 60)

# Mount Google Drive
from google.colab import drive
import os
from pathlib import Path
import shutil

drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# Create data directory structure in Colab
data_dir = Path('data')
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(f'data/{split}/{label}', exist_ok=True)

print("✅ Data directories created!")

# IMPORTANT: Instructions for uploading to Google Drive
print("\n" + "=" * 60)
print("📋 INSTRUCTIONS TO UPLOAD YOUR VIDEOS TO GOOGLE DRIVE:")
print("=" * 60)
print("""
1. On your local computer, go to Google Drive
2. Create a folder called 'Deepfake' in your Drive (or use existing)
3. Inside 'Deepfake', create this structure:
   Deepfake/
   ├── data/
   │   ├── train/
   │   │   ├── real/  (put your real training videos here)
   │   │   └── fake/  (put your fake training videos here)
   │   ├── val/
   │   │   ├── real/  (put your real validation videos here)
   │   │   └── fake/  (put your fake validation videos here)
   │   └── test/
   │       ├── real/  (put your real test videos here)
   │       └── fake/  (put your fake test videos here)

4. Upload all your video files to the appropriate folders
5. Once uploaded, update the DRIVE_DATA_PATH below to match your Drive path
""")

# Configure your Google Drive data path
# Default: /content/drive/MyDrive/Deepfake/data
# Change this if your data is in a different location
DRIVE_DATA_PATH = '/content/drive/MyDrive/Deepfake/data'

# Check if data exists in Drive
if os.path.exists(DRIVE_DATA_PATH):
    print(f"\n✅ Found data in Drive at: {DRIVE_DATA_PATH}")
    
    # Copy data from Drive to Colab
    print("📦 Copying data from Google Drive to Colab...")
    
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
                
                print(f"  ✅ Copied {len(video_files)} videos from {split}/{label}")
            else:
                print(f"  ⚠️  No videos found in {drive_folder}")
    
    print("\n✅ Data copying complete!")
else:
    print(f"\n⚠️  Data not found at: {DRIVE_DATA_PATH}")
    print("Please:")
    print("1. Upload your data folder to Google Drive")
    print("2. Update DRIVE_DATA_PATH variable above to match your Drive path")
    print("3. Re-run this cell")

# Verify data structure
print("\n📊 Data Summary:")
for split in ['train', 'val', 'test']:
    split_dir = Path('data') / split
    if split_dir.exists():
        for label in ['real', 'fake']:
            label_dir = split_dir / label
            if label_dir.exists():
                video_count = len(list(label_dir.glob('*.mp4'))) + \
                             len(list(label_dir.glob('*.avi'))) + \
                             len(list(label_dir.glob('*.mov'))) + \
                             len(list(label_dir.glob('*.mkv')))
                print(f"  {split}/{label}: {video_count} videos")

print("\n" + "=" * 60)
print("STEP 7: Preparing Dataset Metadata")
print("=" * 60)

# ============================================================================
# STEP 7: PREPARE DATASET METADATA
# ============================================================================

import json
from pathlib import Path

def create_metadata(data_dir, split):
    """Create metadata JSON for dataset split."""
    data_path = Path(data_dir)
    split_path = data_path / split
    
    samples = []
    
    # Add real samples
    real_path = split_path / 'real'
    if real_path.exists():
        for video_file in real_path.glob('*.mp4'):
            samples.append({
                'video_path': str(video_file),
                'label': 0  # 0 = real
            })
        for video_file in real_path.glob('*.avi'):
            samples.append({
                'video_path': str(video_file),
                'label': 0
            })
        for video_file in real_path.glob('*.mov'):
            samples.append({
                'video_path': str(video_file),
                'label': 0
            })
        for video_file in real_path.glob('*.mkv'):
            samples.append({
                'video_path': str(video_file),
                'label': 0
            })
    
    # Add fake samples
    fake_path = split_path / 'fake'
    if fake_path.exists():
        for video_file in fake_path.glob('*.mp4'):
            samples.append({
                'video_path': str(video_file),
                'label': 1  # 1 = fake
            })
        for video_file in fake_path.glob('*.avi'):
            samples.append({
                'video_path': str(video_file),
                'label': 1
            })
        for video_file in fake_path.glob('*.mov'):
            samples.append({
                'video_path': str(video_file),
                'label': 1
            })
        for video_file in fake_path.glob('*.mkv'):
            samples.append({
                'video_path': str(video_file),
                'label': 1
            })
    
    return samples

# Generate metadata for all splits
data_dir = 'data'
for split in ['train', 'val', 'test']:
    print(f"\nProcessing {split} split...")
    samples = create_metadata(data_dir, split)
    
    if len(samples) == 0:
        print(f"  ⚠️  Warning: No samples found for {split} split")
        continue
    
    # Save metadata
    metadata_file = Path(data_dir) / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Count labels
    real_count = sum(1 for s in samples if s['label'] == 0)
    fake_count = sum(1 for s in samples if s['label'] == 1)
    
    print(f"  ✅ Created metadata for {len(samples)} samples")
    print(f"     Real: {real_count}, Fake: {fake_count}")
    print(f"     Saved to: {metadata_file}")

print("\n✅ Dataset metadata preparation complete!")

# ============================================================================
# STEP 8: UPDATE CONFIG FOR COLAB
# ============================================================================

print("\n" + "=" * 60)
print("STEP 8: Updating Config for Colab GPU")
print("=" * 60)

import yaml

config_path = 'experiments/configs/default.yaml'

# First, we need to clone the repository or upload the config file
# If you haven't cloned the repo yet, do it now:
print("\n📥 Cloning repository from GitHub...")
print("(If you haven't pushed to GitHub yet, you can upload the config file manually)")

# Uncomment and update with your GitHub repo URL:
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# !mv YOUR_REPO_NAME/* .
# !mv YOUR_REPO_NAME/.* . 2>/dev/null || true

# Or create the config file structure if it doesn't exist
os.makedirs('experiments/configs', exist_ok=True)

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for Colab GPU (can use larger batch size)
    if 'training' in config:
        config['training']['batch_size'] = 2  # Increase from 1 for GPU
        config['training']['num_workers'] = 0  # Set to 0 to avoid MediaPipe multiprocessing segfault
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Config updated for Colab GPU!")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Num workers: {config['training']['num_workers']}")
else:
    print("⚠️  Config file not found. Creating default config...")
    # Create default config
    default_config = {
        'data': {
            'data_dir': 'data',
            'num_frames': 16,
            'frame_size': [224, 224],
            'fps': 25.0
        },
        'model': {
            'video_backbone': 'efficientnet_b0',
            'video_embedding_dim': 512,
            'n_mels': 128,
            'audio_embedding_dim': 512,
            'lip_embedding_dim': 256,
            'fusion_dim': 512,
            'num_heads': 8,
            'num_transformer_layers': 4,
            'dropout': 0.1,
            'use_cross_attention': True
        },
        'training': {
            'batch_size': 2,
            'epochs': 5,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'num_workers': 2,
            'checkpoint_dir': 'experiments/checkpoints',
            'use_focal_loss': False,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'auxiliary_loss_weight': 0.1,
            'scheduler': 'cosine',
            'min_lr': 0.000001,
            'grad_clip': 1.0,
            'early_stopping': True,
            'patience': 10
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print("✅ Default config created!")

# ============================================================================
# STEP 9: INSTALL DEPENDENCIES AND RUN TRAINING
# ============================================================================

print("\n" + "=" * 60)
print("STEP 9: Installing Dependencies and Running Training")
print("=" * 60)

print("\n📦 Installing required packages...")
print("(This may take a few minutes)")

# Install PyYAML if not already installed
try:
    import yaml
except ImportError:
    !pip install pyyaml

# Install other dependencies
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

print("\n✅ Dependencies installed!")

print("\n🚀 Starting training...")
print("(This will take a while depending on your dataset size)")

# Run training
!python src/models/train.py --config experiments/configs/default.yaml

print("\n✅ Training complete!")

# ============================================================================
# STEP 10: DOWNLOAD RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("STEP 10: Downloading Results")
print("=" * 60)

from google.colab import files

# Download best model checkpoint
checkpoint_path = 'experiments/checkpoints/checkpoint_best.pth'
if os.path.exists(checkpoint_path):
    print(f"\n📥 Downloading best checkpoint: {checkpoint_path}")
    files.download(checkpoint_path)
    print("✅ Best checkpoint downloaded!")
else:
    print(f"⚠️  Checkpoint not found at: {checkpoint_path}")

# Download latest checkpoint
latest_checkpoint = 'experiments/checkpoints/checkpoint_latest.pth'
if os.path.exists(latest_checkpoint):
    print(f"\n📥 Downloading latest checkpoint: {latest_checkpoint}")
    files.download(latest_checkpoint)
    print("✅ Latest checkpoint downloaded!")

# Optionally download logs
logs_dir = 'experiments/checkpoints/logs'
if os.path.exists(logs_dir):
    print("\n📦 Creating logs archive...")
    !zip -r results_logs.zip experiments/checkpoints/logs
    files.download('results_logs.zip')
    print("✅ Logs downloaded!")

print("\n" + "=" * 60)
print("🎉 ALL STEPS COMPLETE!")
print("=" * 60)
print("\nYour trained model checkpoints have been downloaded.")
print("You can now use these checkpoints for inference on your local machine.")

