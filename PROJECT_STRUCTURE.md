# Project Structure (Windows)

> **Windows OS Optimized** - All commands are for Windows PowerShell.

```
Deepfake/
├── data/                          # Dataset storage
│   ├── train/                     # Training videos
│   │   ├── real/                  # Real videos
│   │   └── fake/                  # Fake videos
│   ├── val/                       # Validation videos
│   └── test/                       # Test videos
│
├── notebooks/                     # Jupyter notebooks for EDA
│
├── src/                           # Source code
│   ├── preprocessing/             # Data preprocessing
│   │   ├── __init__.py
│   │   ├── face_extractor.py      # Face detection and extraction
│   │   ├── audio_extract.py       # Audio extraction and spectrogram
│   │   └── landmarks.py           # Facial landmark extraction
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── video_stream.py        # Video stream model
│   │   ├── audio_stream.py        # Audio stream model
│   │   ├── lip_sync.py            # Lip-sync module
│   │   ├── fusion_module.py       # Fusion module
│   │   ├── complete_model.py     # Complete model
│   │   └── train.py               # Training script
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset class
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── explainability.py      # Grad-CAM and saliency
│   │
│   └── inference/                 # Inference pipeline
│       ├── __init__.py
│       └── pipeline.py            # Inference pipeline
│
├── demo/                          # Demo application
│   ├── app.py                     # FastAPI application
│   └── templates/
│       └── index.html             # Web interface
│
├── experiments/                   # Experiments and results
│   ├── configs/
│   │   └── default.yaml           # Default configuration
│   ├── checkpoints/               # Model checkpoints
│   └── logs/                      # Training logs
│
├── scripts/                        # Utility scripts
│   ├── prepare_dataset.py         # Dataset preparation
│   └── evaluate.py                # Model evaluation
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
└── PROJECT_STRUCTURE.md            # This file
```

## Quick Start (Windows)

1. **Install dependencies:**
   ```powershell
   # Activate virtual environment first
   .venv\Scripts\Activate.ps1
   
   # Install packages
   pip install -r requirements.txt
   ```

2. **Prepare dataset:**
   ```powershell
   # Create directory structure
   .\setup_dataset.ps1
   
   # Generate metadata
   python scripts/prepare_dataset.py --data_dir data --splits train val test
   ```

3. **Train model:**
   ```powershell
   python src/models/train.py --config experiments/configs/default.yaml
   ```

4. **Evaluate model:**
   ```powershell
   python scripts/evaluate.py --checkpoint experiments/checkpoints/checkpoint_best.pth --data_dir data --split test
   ```

5. **Run demo:**
   ```powershell
   cd demo
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## Dataset Organization

Organize your dataset as follows:
```
data/
├── train/
│   ├── real/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── fake/
│       ├── video1.mp4
│       └── video2.mp4
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

The `prepare_dataset.py` script will automatically create metadata JSON files for each split.

