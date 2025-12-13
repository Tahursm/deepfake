# Deepfake Audio + Video Detection System

> **Windows OS Optimized** - This documentation is optimized for Windows PowerShell.

A production-ready multi-stream deep learning system that detects deepfake audio and video simultaneously by fusing visual (face + lip-sync + temporal) and audio (spectrogram) embeddings using attention/transformer-based fusion.

## Features

- **Multi-Stream Architecture**: Visual stream (face + temporal) + Audio stream (spectrogram) + Lip-sync correlation
- **Advanced Fusion**: Attention-based cross-modal fusion with transformer modules
- **Explainability**: Grad-CAM for visual explanations, audio saliency maps
- **Robust Training**: Data augmentation, regularization, domain adaptation strategies
- **Demo Application**: FastAPI-based web interface for real-time inference

## Project Structure

```
deepfake-project/
├── data/                  # Dataset storage and scripts
├── notebooks/             # EDA & prototyping
├── src/
│   ├── preprocessing/     # Data preprocessing modules
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   └── inference/         # Inference pipeline
├── demo/                  # FastAPI demo application
├── experiments/           # Configs, logs, checkpoints
├── requirements.txt
└── README.md
```

## Installation (Windows)

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd Deepfake
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **(Optional) For GPU support, install CUDA-enabled PyTorch:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Dataset Preparation

1. Download datasets (FaceForensics++, DFDC, FakeAVCeleb, etc.)
2. Organize data in `data/` directory
3. Run preprocessing scripts to extract faces and audio features

## Training

```powershell
python src/models/train.py --config experiments/configs/default.yaml
```

## Inference & Demo

Start the FastAPI server:
```powershell
cd demo
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access the web interface at `http://localhost:8000`

## Model Architecture

- **Visual Stream**: ResNet50/EfficientNet-B0 backbone + Transformer encoder for temporal modeling
- **Audio Stream**: CNN + Transformer on Mel-spectrogram features
- **Lip-Sync Module**: Mouth region CNN + temporal transformer for lip-audio correlation
- **Fusion Module**: Cross-modal attention + Transformer decoder + Classification head

## Evaluation Metrics

- ROC-AUC, Accuracy, Precision, Recall, F1-score
- Equal Error Rate (EER)
- Per-manipulation accuracy
- Cross-dataset generalization

## License

[Specify your license]

## Citation

[Add citation if published]

