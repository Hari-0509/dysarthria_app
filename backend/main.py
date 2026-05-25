"""
FastAPI Backend for Dysarthria Detection
Loads CatBoost models directly and serves predictions
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from catboost import CatBoostClassifier
import tempfile
import logging

# ========================
# Logging
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# FastAPI App
# ========================
app = FastAPI(
    title="Dysarthria Detection API",
    description="AI-powered dysarthria detection using HuBERT + CatBoost",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Global Variables (Loaded Once at Startup)
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Model paths
MODEL_DIR = Path("models")
PRESENCE_MODEL_PATH = MODEL_DIR / "catboost_presence1.cbm"
SEVERITY_MODEL_PATH = MODEL_DIR / "catboost_severity1.cbm"

# Global models (loaded at startup)
feature_extractor = None
hubert = None
presence_model = None
severity_model = None


@app.on_event("startup")
async def load_models():
    """Load all models at startup"""
    global feature_extractor, hubert, presence_model, severity_model
    
    try:
        logger.info("Loading HuBERT feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        
        logger.info("Loading HuBERT model...")
        hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        ).to(DEVICE)
        hubert.eval()
        
        logger.info("Loading CatBoost presence model...")
        presence_model = CatBoostClassifier()
        presence_model.load_model(str(PRESENCE_MODEL_PATH), format="cbm")
        
        logger.info("Loading CatBoost severity model...")
        severity_model = CatBoostClassifier()
        severity_model.load_model(str(SEVERITY_MODEL_PATH), format="cbm")
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise


# ========================
# Helper Functions
# ========================

def extract_hubert_feature(audio_path):
    """Extract HuBERT embeddings from audio"""
    try:
        wav, sr = librosa.load(audio_path, sr=16000)
        
        if len(wav) < 1600:
            raise ValueError("Audio too short (minimum 100ms required)")
        
        inputs = feature_extractor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = hubert(inputs.input_values.to(DEVICE))
        
        feats = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        # Mean + Std pooling
        mean_feat = feats.mean(axis=0)
        std_feat = feats.std(axis=0)
        
        embedding = np.concatenate([mean_feat, std_feat])
        return embedding
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return None


def predict_audio(audio_path):
    """Predict dysarthria presence and severity"""
    try:
        feat = extract_hubert_feature(audio_path)
        if feat is None:
            return None, None, None
        
        feat = feat.reshape(1, -1)
        
        # Presence prediction
        prob = presence_model.predict_proba(feat)[0][1]
        pred = int(prob >= 0.5)
        
        if pred == 0:
            return "NORMAL", float(round(1 - prob, 3)), None
        
        # Severity prediction
        severity_pred = int(severity_model.predict(feat).flatten()[0])
        
        severity_map = {
            0: "MILD",
            1: "MODERATE",
            2: "SEVERE"
        }
        
        severity = severity_map.get(severity_pred, "UNKNOWN")
        
        return "DYSARTHRIA", float(round(prob, 3)), severity
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None, None, None


# ========================
# API Endpoints
# ========================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": presence_model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze audio file for dysarthria
    
    Returns:
        - prediction: NORMAL or DYSARTHRIA
        - confidence: confidence score (0-1)
        - severity: MILD, MODERATE, or SEVERE (only if DYSARTHRIA)
    """
    try:
        # Validate file
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files accepted")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            content = await file.read()
            temp.write(content)
            audio_path = temp.name
        
        # Predict
        prediction, confidence, severity = predict_audio(audio_path)
        
        if prediction is None:
            raise HTTPException(status_code=400, detail="Audio processing failed")
        
        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "severity": severity,
            "model_version": "1.0"
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    """Extract raw HuBERT embeddings"""
    try:
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files accepted")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            content = await file.read()
            temp.write(content)
            audio_path = temp.name
        
        feat = extract_hubert_feature(audio_path)
        
        if feat is None:
            raise HTTPException(status_code=400, detail="Feature extraction failed")
        
        return {
            "success": True,
            "embedding": feat.tolist(),
            "shape": list(feat.shape),
            "embedding_dim": int(feat.shape[0])
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Feature extraction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
def api_info():
    """API information"""
    return {
        "name": "Dysarthria Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict dysarthria from audio",
            "/extract-features": "Extract HuBERT embeddings",
            "/info": "API information"
        },
        "device": DEVICE,
        "supported_formats": ["wav"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
