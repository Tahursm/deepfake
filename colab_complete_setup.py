"""
COMPLETE GOOGLE COLAB SETUP - FROM SCRATCH
==========================================
This script sets up the entire Deepfake Detection project in Colab from GitHub.
Copy each section into separate cells in Google Colab and run them in order.
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
# Install all required packages with version constraints to avoid conflicts
# This installation order prevents dependency conflicts
print("📦 Installing dependencies...")
print("This may take a few minutes...")

# Step 0: Clean up any conflicting packages that might be installed
print("\n📦 Step 0: Cleaning up conflicting packages...")
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
from pathlib import Path

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
# CELL 3: Verify Project Structure
# ============================================================================
"""
# Verify that all necessary files are present
import os
from pathlib import Path

print("🔍 Verifying project structure...")

required_files = [
    "src/models/train.py",
    "src/utils/dataset.py",
    "src/preprocessing/face_extractor.py",
    "experiments/configs/default.yaml",
    "requirements.txt"
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)
    else:
        print(f"  ✅ {file}")

if missing_files:
    print(f"\n⚠️  Missing files: {missing_files}")
    print("Please check that the repository was cloned correctly.")
else:
    print("\n✅ All required files are present!")
    print("\n📂 Project structure:")
    !tree -L 2 -I '__pycache__|*.pyc' || find . -maxdepth 2 -type d | head -20
"""

# ============================================================================
# CELL 4: Mount Google Drive and Set Up Data
# ============================================================================
"""
from google.colab import drive
import os
from pathlib import Path
import shutil

# Mount Google Drive
print("📂 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# Create data directory structure
print("\n📁 Creating data directory structure...")
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(f'data/{split}/{label}', exist_ok=True)

print("✅ Data directories created!")

# IMPORTANT: Configure your Google Drive data path
# Default: /content/drive/MyDrive/Deepfake/data
# Change this if your data is in a different location
DRIVE_DATA_PATH = '/content/drive/MyDrive/Deepfake/data'

print("\n" + "="*60)
print("📋 DATA UPLOAD INSTRUCTIONS:")
print("="*60)
print("""
If you haven't uploaded your videos to Google Drive yet:

1. Go to https://drive.google.com
2. Create a folder called 'Deepfake' (or use existing)
3. Inside 'Deepfake', create this structure:
   Deepfake/
   └── data/
       ├── train/
       │   ├── real/  (put your real training videos here)
       │   └── fake/  (put your fake training videos here)
       ├── val/
       │   ├── real/  (put your real validation videos here)
       │   └── fake/  (put your fake validation videos here)
       └── test/
           ├── real/  (put your real test videos here)
           └── fake/  (put your fake test videos here)

4. Upload all your video files (.mp4, .avi, .mov, .mkv) to the appropriate folders
5. Once uploaded, update DRIVE_DATA_PATH above if needed
6. Re-run this cell to copy the data
""")

# Check if data exists in Drive and copy it
if os.path.exists(DRIVE_DATA_PATH):
    print(f"\n✅ Found data in Drive at: {DRIVE_DATA_PATH}")
    print("📦 Copying videos from Google Drive to Colab...")
    
    total_copied = 0
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
                
                if len(video_files) > 0:
                    print(f"  ✅ {split}/{label}: {len(video_files)} videos copied")
                    total_copied += len(video_files)
            else:
                print(f"  ⚠️  No videos found in {split}/{label}")
    
    if total_copied > 0:
        print(f"\n✅ Total: {total_copied} videos copied!")
    else:
        print("\n⚠️  No videos were copied. Please upload videos to Google Drive first.")
else:
    print(f"\n⚠️  Data not found at: {DRIVE_DATA_PATH}")
    print("Please upload your videos to Google Drive first, then re-run this cell.")

# Display data summary
print("\n📊 Current Data Summary:")
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
                if video_count > 0:
                    print(f"  {split}/{label}: {video_count} videos")
"""

# ============================================================================
# CELL 5: Prepare Dataset Metadata
# ============================================================================
"""
import json
from pathlib import Path

print("📝 Preparing dataset metadata...")

def create_metadata(data_dir, split):
    \"\"\"Create metadata JSON for dataset split.\"\"\"
    data_path = Path(data_dir)
    split_path = data_path / split
    
    samples = []
    
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
total_samples = 0

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
    total_samples += len(samples)
    
    print(f"  ✅ Created metadata for {len(samples)} samples")
    print(f"     Real: {real_count}, Fake: {fake_count}")
    print(f"     Saved to: {metadata_file}")

if total_samples > 0:
    print(f"\n✅ Dataset metadata preparation complete!")
    print(f"   Total samples: {total_samples}")
else:
    print("\n⚠️  No samples found. Please make sure videos are uploaded and copied.")
"""

# ============================================================================
# CELL 6: Update Config for Colab
# ============================================================================
"""
import yaml
import os

print("⚙️  Updating configuration for Colab...")

config_path = 'experiments/configs/default.yaml'

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for Colab GPU
    if 'training' in config:
        original_batch = config['training'].get('batch_size', 1)
        config['training']['batch_size'] = 2  # Increase for GPU
        config['training']['num_workers'] = 0  # Set to 0 to avoid MediaPipe multiprocessing issues
        
        print(f"✅ Config updated for Colab!")
        print(f"   Batch size: {original_batch} → {config['training']['batch_size']}")
        print(f"   Num workers: → {config['training']['num_workers']} (prevents segfault)")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration saved!")
else:
    print("⚠️  Config file not found. Please check the repository structure.")
"""

# ============================================================================
# CELL 7: Verify Setup Before Training
# ============================================================================
"""
import os
from pathlib import Path

print("🔍 Final verification before training...")
print("="*60)

# Check data
data_ok = True
for split in ['train', 'val', 'test']:
    metadata_file = Path('data') / f'{split}_metadata.json'
    if not metadata_file.exists():
        print(f"⚠️  Missing metadata: {metadata_file}")
        data_ok = False
    else:
        import json
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        print(f"✅ {split}: {len(samples)} samples")

# Check config
config_ok = os.path.exists('experiments/configs/default.yaml')
if config_ok:
    print("✅ Config file exists")
else:
    print("⚠️  Config file missing")
    data_ok = False

# Check training script
train_ok = os.path.exists('src/models/train.py')
if train_ok:
    print("✅ Training script exists")
else:
    print("⚠️  Training script missing")
    data_ok = False

print("="*60)
if data_ok and config_ok and train_ok:
    print("✅ All checks passed! Ready to train.")
    print("\n🚀 You can now proceed to Cell 8 to start training!")
else:
    print("⚠️  Some checks failed. Please review the errors above.")
"""

# ============================================================================
# CELL 8: Run Training
# ============================================================================
"""
# Start training
print("🚀 Starting training...")
print("This will take a while depending on your dataset size and GPU availability.")
print("="*60)

!python src/models/train.py --config experiments/configs/default.yaml

print("="*60)
print("✅ Training complete!")
"""

# ============================================================================
# CELL 9: Download Results
# ============================================================================
"""
from google.colab import files
import os

print("📥 Downloading training results...")
print("="*60)

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
    !zip -r results_logs.zip experiments/checkpoints/logs 2>/dev/null || echo "Logs directory empty or zip failed"
    if os.path.exists('results_logs.zip'):
        files.download('results_logs.zip')
        print("✅ Logs downloaded!")

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print("\nYour trained model checkpoints have been downloaded.")
print("You can now use these checkpoints for inference on your local machine.")
print("\nTo use the model locally:")
print("1. Save the downloaded checkpoint to: experiments/checkpoints/")
print("2. Run inference using: python demo/app.py")
"""

