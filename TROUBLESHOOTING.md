# Troubleshooting Guide (Windows)

> **Windows OS Optimized** - All solutions are for Windows PowerShell.

## Dependency Conflicts

### Protobuf Conflict with mysql-connector-python

**Error:**
```
mysql-connector-python 8.2.0 requires protobuf<=4.21.12,>=4.21.1, 
but you have protobuf 4.25.8 which is incompatible.
```

**Solution 1 (Recommended):** Uninstall mysql-connector-python if you don't need it:
```powershell
pip uninstall mysql-connector-python -y
```

**Solution 2:** Use the fix script:
```powershell
python scripts/fix_dependencies.py
```

**Solution 3:** Manually install compatible protobuf version:
```powershell
pip install 'protobuf>=4.21.0,<4.26.0'
```

**Note:** This warning is usually non-critical. The packages will still work, but you may encounter issues if you actually use mysql-connector-python.

---

## Command Path Errors

### Error: Command not recognized

**Error:**
```
0.e:\Deepfake\.venv\Scripts\python.exe: The term '0.e:\Deepfake\.venv\Scripts\python.exe' is not recognized
```

**Cause:** Typo in the path - `0.e:` instead of `E:`

**Solution:** Use the correct path. In PowerShell:

```powershell
# Activate virtual environment first
.venv\Scripts\Activate.ps1

# Then run commands normally
python -m pip install -r requirements.txt
```

Or use the full correct path:
```powershell
E:\Deepfake\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Virtual Environment Issues

### Activating Virtual Environment

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**If activation fails in PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Installation Issues

### CUDA/GPU Support

If you have an NVIDIA GPU and want CUDA support:

```powershell
# Uninstall CPU-only PyTorch first
pip uninstall torch torchvision torchaudio -y

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### FFmpeg Not Found

If you get errors about FFmpeg:

**Windows Installation:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH:
   ```powershell
   # Add to system PATH (requires admin)
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "Machine")
   ```
4. Or install via Chocolatey (if installed):
   ```powershell
   choco install ffmpeg
   ```
5. Restart PowerShell after adding to PATH

---

## Import Errors

### "No module named 'src'"

**Solution:** Run from project root directory, or add to PYTHONPATH:

**Windows PowerShell:**
```powershell
# Temporary (current session only)
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Or permanent (for current user)
[Environment]::SetEnvironmentVariable("PYTHONPATH", "$PWD;$env:PYTHONPATH", "User")
```

Or install in development mode:
```powershell
pip install -e .
```

---

## Model Loading Errors

### "Model checkpoint not found"

**Error:** When running demo, model checkpoint is missing.

**Solution:**
1. Train a model first:
   ```powershell
   python src/models/train.py --config experiments/configs/default.yaml
   ```

2. Or update the model path in `demo/app.py`:
   ```python
   model_path = Path("path/to/your/checkpoint.pth")
   ```

---

## Memory Issues

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in `experiments/configs/default.yaml`:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
   ```

2. Use gradient accumulation:
   ```python
   # In train.py, accumulate gradients over multiple batches
   ```

3. Use smaller model:
   ```yaml
   model:
     video_backbone: "efficientnet_b0"  # Instead of resnet50
   ```

4. Process fewer frames:
   ```yaml
   data:
     num_frames: 32  # Instead of 64
   ```

---

## Face Detection Issues

### No faces detected in videos

**Solutions:**
1. Check video quality - ensure faces are clearly visible
2. Adjust detection confidence in `src/preprocessing/face_extractor.py`:
   ```python
   FaceExtractor(min_detection_confidence=0.3)  # Lower threshold
   ```
3. Ensure good lighting in videos
4. Check if videos contain faces at all

---

## Audio Processing Issues

### Librosa errors

**Error:** `librosa.load()` fails

**Solutions:**
1. Ensure FFmpeg is installed (see above)
2. Check audio codec compatibility
3. Try converting video to standard format:
   ```powershell
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
   ```

---

## Training Issues

### Loss is NaN

**Solutions:**
1. Check for corrupted data
2. Reduce learning rate
3. Add gradient clipping (already in config)
4. Check data normalization

### Training is too slow

**Solutions:**
1. Use GPU if available
2. Reduce number of frames
3. Use smaller batch size
4. Enable data caching (already implemented)
5. Use fewer workers if CPU-bound

---

## Demo Application Issues

### Error: "Analysis failed" or "Error analyzing video"

**Common Causes and Solutions:**

1. **Model not loaded:**
   - Check if model checkpoint exists: `experiments\checkpoints\checkpoint_best.pth`
   - If missing, train a model first: `python src/models/train.py --config experiments/configs/default.yaml`
   - Check server logs for model loading errors

2. **No faces detected in video:**
   - Error message: "No faces detected in the video"
   - **Solution:** Upload a video with clear, visible faces. Ensure good lighting and face is clearly visible

3. **Video file format issues:**
   - Error message: "Unsupported file format" or "Could not extract frames"
   - **Solution:** Use supported formats: MP4, AVI, MOV, MKV, WebM
   - Convert video if needed: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4`

4. **Audio extraction failed:**
   - Error message: "Could not extract audio from video"
   - **Solution:** 
     - Ensure video has an audio track
     - Install FFmpeg: `choco install ffmpeg` or download from https://ffmpeg.org
     - Check if librosa can read the file: `python -c "import librosa; librosa.load('your_video.mp4')"`

5. **Missing dependencies:**
   ```powershell
   # Check if all required packages are installed
   pip install torch torchvision opencv-python mediapipe librosa soundfile numpy
   ```

6. **Check server status:**
   ```powershell
   # Visit http://localhost:8000/status in browser
   # Should return: {"status": "ready", "model_loaded": true}
   ```

7. **View detailed error logs:**
   - Check the terminal/console where the demo server is running
   - Look for Python traceback errors
   - The improved error handling now shows specific error messages

**Debugging Steps:**
```powershell
# 1. Check if model exists
Test-Path "experiments\checkpoints\checkpoint_best.pth"

# 2. Test video file
python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); cap.release()"

# 3. Test face detection
python -c "import cv2; import mediapipe as mp; mp_face = mp.solutions.face_detection; detector = mp_face.FaceDetection(); img = cv2.imread('test.jpg'); results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); print('Faces found:', len(results.detections) if results.detections else 0)"
```

### Port already in use

**Error:** `Address already in use`

**Solution:** Use a different port:
```powershell
uvicorn app:app --reload --port 8001
```

### CORS errors

If accessing from different origin, add CORS middleware in `demo/app.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Getting Help

If you encounter other issues:

1. Check the error message carefully
2. Search for the error online
3. Check Python and package versions
4. Ensure all dependencies are installed
5. Try creating a fresh virtual environment

For project-specific issues, check:
- `README.md` for general information
- `QUICKSTART.md` for setup instructions
- `PROJECT_STRUCTURE.md` for project layout

