"""
FastAPI demo application for deepfake detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import torch
from pathlib import Path
import tempfile
import shutil
from typing import Optional
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import DeepfakeInferencePipeline

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Model pipeline (loaded on startup)
pipeline: Optional[DeepfakeInferencePipeline] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global pipeline
    
    # Model path - adjust based on your checkpoint location
    model_path = Path(__file__).parent.parent / "experiments" / "checkpoints" / "checkpoint_best.pth"
    
    print(f"Looking for model at: {model_path.absolute()}")
    
    if not model_path.exists():
        print(f"❌ ERROR: Model checkpoint not found at {model_path.absolute()}")
        print(f"   Expected path: {model_path.absolute()}")
        print("\n📋 To fix this:")
        print("   1. Make sure you've downloaded checkpoint_best.pth from Colab")
        print("   2. Place it in: experiments/checkpoints/checkpoint_best.pth")
        print("   3. Restart the server")
        pipeline = None
    else:
        try:
            print(f"✅ Checkpoint found! Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
            print("🔄 Loading model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   Using device: {device}")
            
            # Load checkpoint to inspect config
            checkpoint = torch.load(model_path, map_location='cpu')
            checkpoint_config = checkpoint.get('config', {})
            
            # Load config to get num_frames
            config_path = Path(__file__).parent.parent / "experiments" / "configs" / "default.yaml"
            num_frames = 16  # default
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    num_frames = config.get('data', {}).get('num_frames', 16)
            
            # Check what video_backbone is in checkpoint
            model_config_in_checkpoint = checkpoint_config.get('model', {})
            video_backbone = model_config_in_checkpoint.get('video_backbone', 'unknown')
            print(f"   Checkpoint video_backbone: {video_backbone}")
            print(f"   Using num_frames: {num_frames}")
            
            # Check if checkpoint has the config
            if not checkpoint_config:
                print("   ⚠️  Warning: Checkpoint doesn't have config, will use defaults")
            
            pipeline = DeepfakeInferencePipeline(
                model_path=str(model_path),
                device=device,
                num_frames=num_frames,
                frame_size=(224, 224)
            )
            print(f"✅ Model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            print("\n📋 Possible causes:")
            print("   1. Checkpoint file might be corrupted")
            print("   2. Missing dependencies (check requirements.txt)")
            print("   3. Model architecture mismatch")
            pipeline = None


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to avoid 404 errors."""
    return Response(status_code=204)  # No Content


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_explanations: bool = False
):
    """
    Predict if uploaded video is deepfake.
    
    Args:
        file: Video file to analyze
        return_explanations: Return explainability visualizations
    
    Returns:
        Prediction results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Validate file
        if not file.filename:
            raise ValueError("No file uploaded")
        
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if Path(file.filename).suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}")
        
        # Run prediction
        result = pipeline.predict(
            video_path=tmp_path,
            return_explanations=return_explanations
        )
        
        # Convert numpy arrays to lists for JSON serialization
        if 'gradcam_frames' in result:
            # Save gradcam frames as base64 or file paths
            # For simplicity, we'll just return metadata
            result['gradcam_available'] = True
            del result['gradcam_frames']  # Remove large arrays
        
        if 'audio_saliency' in result:
            # Convert to list for JSON
            result['audio_saliency'] = result['audio_saliency'].tolist()
        
        return JSONResponse(content=result)
    
    except ValueError as e:
        # User-friendly error messages
        error_msg = str(e)
        if "No frames extracted" in error_msg:
            error_msg = "Could not extract frames from video. Please check if the video file is valid."
        elif "No faces extracted" in error_msg:
            error_msg = "No faces detected in the video. Please upload a video with clear, visible faces."
        elif "Error extracting audio" in error_msg:
            error_msg = "Could not extract audio from video. The video may be corrupted or have no audio track."
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        import traceback
        error_detail = str(e)
        print(f"Prediction error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_detail}")
    
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/status")
async def status():
    """Check API status."""
    model_path = Path(__file__).parent.parent / "experiments" / "checkpoints" / "checkpoint_best.pth"
    return {
        "status": "ready" if pipeline is not None else "not_ready",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else None,
        "checkpoint_exists": model_path.exists(),
        "checkpoint_path": str(model_path.absolute()) if model_path.exists() else None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

