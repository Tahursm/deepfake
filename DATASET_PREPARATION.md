# Dataset Preparation Guide (Windows)

> **Windows OS Optimized** - All commands are for Windows PowerShell.

This guide explains how to prepare your dataset for the deepfake detection system.

## Directory Structure

Organize your videos in the following structure:

```
data/
├── train/
│   ├── real/          # Real/authentic videos
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── fake/          # Deepfake/manipulated videos
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── val/               # Validation set
│   ├── real/
│   │   └── ...
│   └── fake/
│       └── ...
└── test/              # Test set
    ├── real/
    │   └── ...
    └── fake/
        └── ...
```

## Step-by-Step Instructions

### Step 1: Create Directory Structure

**Windows PowerShell:**
```powershell
# Navigate to project directory
cd E:\Deepfake

# Create directory structure
New-Item -ItemType Directory -Force -Path "data\train\real"
New-Item -ItemType Directory -Force -Path "data\train\fake"
New-Item -ItemType Directory -Force -Path "data\val\real"
New-Item -ItemType Directory -Force -Path "data\val\fake"
New-Item -ItemType Directory -Force -Path "data\test\real"
New-Item -ItemType Directory -Force -Path "data\test\fake"
```

**Or use the automated script:**
```powershell
.\setup_dataset.ps1
```

### Step 2: Organize Your Videos

1. **Collect your videos:**
   - Real videos: Authentic, unmanipulated videos
   - Fake videos: Deepfake/manipulated videos

2. **Split into train/val/test:**
   - **Train**: 70-80% of your data (for training)
   - **Val**: 10-15% (for validation during training)
   - **Test**: 10-15% (for final evaluation)

3. **Copy videos to appropriate folders:**
   
   **Important:** Replace `"C:\path\to\real\videos\"` and `"C:\path\to\fake\videos\"` with the actual paths where your original videos are stored. Your source videos can be located anywhere on your system (e.g., `"C:\Users\YourName\Downloads\real_videos\"`, `"D:\Datasets\FaceForensics\real\"`, etc.).
   
   ```powershell
   # Example: Copy real videos to train/real
   # Replace "C:\path\to\real\videos\" with your actual source directory
   Copy-Item "C:\path\to\real\videos\*.mp4" "data\train\real\"
   
   # Example: Copy fake videos to train/fake
   # Replace "C:\path\to\fake\videos\" with your actual source directory
   Copy-Item "C:\path\to\fake\videos\*.mp4" "data\train\fake\"
   
   # Example: Copy real videos to val/real
   Copy-Item "C:\path\to\real\videos\*.mp4" "data\val\real\"
   
   # Example: Copy fake videos to val/fake
   Copy-Item "C:\path\to\fake\videos\*.mp4" "data\val\fake\"
   
   # Example: Copy real videos to test/real
   Copy-Item "C:\path\to\real\videos\*.mp4" "data\test\real\"
   
   # Example: Copy fake videos to test/fake
   Copy-Item "C:\path\to\fake\videos\*.mp4" "data\test\fake\"
   ```
   
   **Note:** Make sure you've already split your videos into train/val/test sets before copying. You should copy different subsets of your videos to each split (train, val, test), not the same videos to all three.

### Step 3: Video Format Requirements

**Supported formats:**
- MP4 (recommended)
- AVI
- MOV
- Other formats supported by OpenCV

**Video requirements:**
- Should contain faces (preferably clear, well-lit faces)
- Should have audio track (for audio+video analysis)
- Recommended duration: 3-10 seconds per clip
- Resolution: Any (will be resized during preprocessing)

**Tips:**
- Higher quality videos generally work better
- Ensure faces are clearly visible
- Good lighting helps with face detection
- Videos with multiple people: The system will detect the primary face

### Step 4: Generate Metadata

After organizing your videos, generate metadata files:

```powershell
# Activate virtual environment (if not already activated)
.venv\Scripts\Activate.ps1

# Run the preparation script
python scripts/prepare_dataset.py --data_dir data --splits train val test
```

This will create:
- `data/train_metadata.json`
- `data/val_metadata.json`
- `data/test_metadata.json`

These files contain the list of videos and their labels.

### Step 5: Verify Dataset

Check that metadata files were created:

```powershell
# List metadata files
Get-ChildItem data\*_metadata.json

# View sample metadata (first few lines)
Get-Content data\train_metadata.json | Select-Object -First 10
```

## Example: Using Public Datasets

### FaceForensics++

1. **Download FaceForensics++:**
   - Visit: https://github.com/ondyari/FaceForensics
   - Download the dataset

2. **Extract videos:**
   ```powershell
   # Extract to a temporary location
   # Then organize into data/ structure
   ```

3. **Organize:**
   - Real videos → `data/train/real/`, `data/val/real/`, `data/test/real/`
   - Fake videos (DeepFakes, Face2Face, etc.) → `data/train/fake/`, etc.

### DFDC (DeepFake Detection Challenge)

1. **Download DFDC:**
   - Visit: https://www.kaggle.com/c/deepfake-detection-challenge
   - Download train/test videos

2. **Organize:**
   - Use the provided metadata to split into real/fake
   - Organize into train/val/test splits

### FakeAVCeleb

1. **Download FakeAVCeleb:**
   - Visit: https://github.com/DASH-Lab/FakeAVCeleb
   - Download the dataset

2. **Organize:**
   - Already organized by real/fake
   - Split into train/val/test manually or using a script

## Dataset Statistics

After preparation, check your dataset:

```powershell
# Count videos in each split
Write-Host "Train - Real: $((Get-ChildItem data\train\real\*.mp4).Count)"
Write-Host "Train - Fake: $((Get-ChildItem data\train\fake\*.mp4).Count)"
Write-Host "Val - Real: $((Get-ChildItem data\val\real\*.mp4).Count)"
Write-Host "Val - Fake: $((Get-ChildItem data\val\fake\*.mp4).Count)"
Write-Host "Test - Real: $((Get-ChildItem data\test\real\*.mp4).Count)"
Write-Host "Test - Fake: $((Get-ChildItem data\test\fake\*.mp4).Count)"
```

## Recommended Dataset Sizes

**Minimum (for testing):**
- Train: 100-200 videos per class
- Val: 20-50 videos per class
- Test: 20-50 videos per class

**Good (for decent results):**
- Train: 1000+ videos per class
- Val: 100+ videos per class
- Test: 100+ videos per class

**Excellent (for production):**
- Train: 10,000+ videos per class
- Val: 1000+ videos per class
- Test: 1000+ videos per class

## Troubleshooting

### No videos found

**Error:** "Warning: No samples found for train split"

**Solution:**
- Check that videos are in the correct directories
- Verify video file extensions (.mp4, .avi, etc.)
- Ensure directory names are exactly `real` and `fake` (lowercase)

### Videos not loading

**Error:** "Error extracting audio from video"

**Solution:**
- Install FFmpeg (see TROUBLESHOOTING.md)
- Convert videos to standard format:
  ```powershell
  ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
  ```

### Imbalanced dataset

If you have more real than fake videos (or vice versa):

1. **Option 1:** Use data augmentation (already implemented)
2. **Option 2:** Use focal loss (enable in config)
3. **Option 3:** Collect more videos for the minority class
4. **Option 4:** Use class weights in training

## Quick Start Script

Save this as `setup_dataset.ps1`:

```powershell
# Create directory structure
$splits = @("train", "val", "test")
$classes = @("real", "fake")

foreach ($split in $splits) {
    foreach ($class in $classes) {
        $path = "data\$split\$class"
        New-Item -ItemType Directory -Force -Path $path | Out-Null
        Write-Host "Created: $path"
    }
}

Write-Host "`nDirectory structure created!"
Write-Host "Now copy your videos to:"
Write-Host "  - data\train\real\  (real videos for training)"
Write-Host "  - data\train\fake\  (fake videos for training)"
Write-Host "  - data\val\real\    (real videos for validation)"
Write-Host "  - data\val\fake\    (fake videos for validation)"
Write-Host "  - data\test\real\   (real videos for testing)"
Write-Host "  - data\test\fake\  (fake videos for testing)"
Write-Host "`nThen run: python scripts/prepare_dataset.py --data_dir data --splits train val test"
```

Run it:
```powershell
.\setup_dataset.ps1
```

## Next Steps

After preparing your dataset:

1. **Verify metadata:**
   ```powershell
   python scripts/prepare_dataset.py --data_dir data --splits train val test
   ```

2. **Start training:**
   ```powershell
   python src/models/train.py --config experiments/configs/default.yaml
   ```

3. **Monitor progress:**
   - Check TensorBoard logs in `experiments/checkpoints/logs/`
   - Or use Weights & Biases if configured

