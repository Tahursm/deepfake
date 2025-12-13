# Quick Start Guide (Windows)

> **Windows OS Optimized** - All commands are for Windows PowerShell.

## Installation

1. **Clone and setup:**
   ```powershell
   cd E:\Deepfake
   
   # Activate virtual environment
   .venv\Scripts\Activate.ps1
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **For GPU support (optional):**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Dataset Preparation

### Quick Setup (Windows PowerShell)

1. **Create directory structure:**
   ```powershell
   .\setup_dataset.ps1
   ```
   Or manually:
   ```powershell
   New-Item -ItemType Directory -Force -Path "data\train\real"
   New-Item -ItemType Directory -Force -Path "data\train\fake"
   New-Item -ItemType Directory -Force -Path "data\val\real"
   New-Item -ItemType Directory -Force -Path "data\val\fake"
   New-Item -ItemType Directory -Force -Path "data\test\real"
   New-Item -ItemType Directory -Force -Path "data\test\fake"
   ```

2. **Organize your videos:**
   - Copy **real/authentic videos** to:
     - `data/train/real/`
     - `data/val/real/`
     - `data/test/real/`
   - Copy **fake/manipulated videos** to:
     - `data/train/fake/`
     - `data/val/fake/`
     - `data/test/fake/`

3. **Generate metadata:**
   ```powershell
   python scripts/prepare_dataset.py --data_dir data --splits train val test
   ```

**For detailed instructions, see `DATASET_PREPARATION.md`**

## Training

1. **Configure training (optional):**
   Edit `experiments/configs/default.yaml` to adjust hyperparameters.

2. **Start training:**
   ```powershell
   python src/models/train.py --config experiments/configs/default.yaml
   ```

3. **Resume training (if interrupted):**
   ```powershell
   python src/models/train.py --config experiments/configs/default.yaml --resume
   ```

## Evaluation

```powershell
python scripts/evaluate.py `
    --checkpoint experiments/checkpoints/checkpoint_best.pth `
    --data_dir data `
    --split test
```

## Demo Application

1. **Start the FastAPI server:**
   ```powershell
   cd demo
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open in browser:**
   Navigate to `http://localhost:8000`

3. **Upload a video:**
   - Click or drag & drop a video file
   - Click "Analyze Video"
   - View prediction results

## Model Architecture

- **Video Stream**: ResNet50/EfficientNet-B0 + Transformer encoder
- **Audio Stream**: CNN + Transformer on Mel-spectrogram
- **Lip-Sync Module**: Mouth region CNN + temporal transformer
- **Fusion Module**: Cross-modal attention + Transformer decoder

## Key Features

✅ Multi-stream architecture (video + audio + lip-sync)  
✅ Attention-based fusion  
✅ Transformer temporal modeling  
✅ Explainability (Grad-CAM, audio saliency)  
✅ Data augmentation  
✅ Focal loss for class imbalance  
✅ Early stopping & learning rate scheduling  

## Troubleshooting

### Dependency Conflicts

**Protobuf conflict warning:**
```powershell
# If you see mysql-connector-python conflict, uninstall it (not needed):
pip uninstall mysql-connector-python -y

# Or run the fix script:
python scripts/fix_dependencies.py
```

**Command path errors:**
- Make sure to activate virtual environment first: `.venv\Scripts\Activate.ps1`
- Use correct path: `E:\Deepfake\.venv\Scripts\python.exe` (not `0.e:`)

### Common Issues

**Issue: "No module named 'src'"**
- Make sure you're running from the project root directory (`E:\Deepfake`)
- Or add the project root to PYTHONPATH:
  ```powershell
  $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
  ```

**Issue: "Model checkpoint not found"**
- Train a model first using the training script
- Or update the model path in `demo/app.py`

**Issue: CUDA out of memory**
- Reduce batch size in config file
- Use gradient accumulation
- Use smaller model (EfficientNet-B0 instead of ResNet50)

**Issue: No faces detected**
- Check video quality and lighting
- Ensure faces are clearly visible
- Try adjusting face detection confidence threshold

**For more troubleshooting help, see `TROUBLESHOOTING.md`**

## Next Steps

1. Download datasets (FaceForensics++, DFDC, FakeAVCeleb)
2. Train on your dataset
3. Evaluate on test set
4. Deploy using the demo application
5. Experiment with different architectures and hyperparameters

