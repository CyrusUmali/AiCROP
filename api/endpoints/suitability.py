from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import os
import json
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Updated paths for individual models
MODELS_DIR = Path("models/individual")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables to store loaded models and metadata
models: Dict[str, dict] = {}  # Now each entry will contain model data, not just the model
label_encoder = None
feature_columns = None
crop_classes = None
df = None  # Dataset for reference

def load_model(model_name: str) -> dict:
    """Load a specific model from disk"""
    model_filename = model_name.lower().replace(' ', '_') + '.pkl'
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_all_models() -> Dict[str, dict]:
    """Load all available models"""
    available_models = {}
    
    # Get all .pkl files in models directory (excluding metrics files)
    for file_path in MODELS_DIR.glob("*.pkl"):
        if "_metrics" in file_path.name:
            continue
            
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            model_name = model_data.get('model_name', file_path.stem)
            available_models[model_name] = model_data
    
    return available_models

def initialize_system():
    """Initialize the system by loading models and metadata"""
    global models, label_encoder, feature_columns, crop_classes, df
    
    try:
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Load model summary to get available crops and features
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        crop_classes = summary['dataset_info']['crops']
        feature_columns = summary['dataset_info']['features']
        
        # Load all models
        models = load_all_models()
        
        if not models:
            raise RuntimeError("No models found in models/individual directory")
        
        # Extract label encoder from first model (all should have the same)
        first_model_data = next(iter(models.values()))
        label_encoder = first_model_data.get('label_encoder')
        
        if label_encoder is None:
            raise RuntimeError("Label encoder not found in model data")
        
        print(f"✓ System initialized with {len(models)} models")
        print(f"✓ Available crops: {len(crop_classes)}")
        print(f"✓ Available models: {list(models.keys())}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize system: {str(e)}")

# Initialize on startup
try:
    initialize_system()
except Exception as e:
    print(f"⚠ Warning: Failed to initialize: {e}")
    # System will fail on first request

class CropSuitabilityRequest(BaseModel):
    """Request model for crop suitability check"""
    soil_ph: float = Field(..., description="Soil pH value")
    fertility_ec: float = Field(..., description="Fertility/EC (Electrical Conductivity) in dS/m")
    humidity: float = Field(..., description="Relative humidity in %")
    sunlight: float = Field(..., description="Sunlight intensity in hours/day or lux")
    soil_temp: float = Field(..., description="Soil temperature in °C")
    soil_moisture: float = Field(..., description="Soil moisture content in %")
    crop: str = Field(..., description="Crop to check suitability for")
    selected_models: Optional[List[str]] = Field(
        None, 
        description="List of model names to use for prediction. If None, uses all available models."
    )

class ParameterAnalysis(BaseModel):
    """Model for parameter analysis results"""
    status: str  # 'low', 'optimal', or 'high'
    current: float
    ideal_min: float
    ideal_max: float
    difference: Optional[float] = None

class SuitabilityResponse(BaseModel):
    """Response model for crop suitability check"""
    is_suitable: bool
    confidence: float
    crop: str
    image_url: Optional[str] = None
    parameters_analysis: Dict[str, ParameterAnalysis]
    model_used: List[str]
    model_confidence: Dict[str, float]  # New: confidence per model
    disclaimer: str = "Results should be verified with local agricultural experts"

def analyze_parameters(input_data: pd.DataFrame, crop: str) -> Dict[str, ParameterAnalysis]:
    """
    Compare input parameters with ideal ranges for the specified crop
    """
    if df is None:
        return {}
        
    crop_data = df[df['label'] == crop]
    if len(crop_data) == 0:
        return {}
    
    analysis = {}
    for param in feature_columns:
        ideal_min = crop_data[param].quantile(0.01)
        ideal_max = crop_data[param].quantile(0.99)
        current_val = input_data[param].values[0]
        
        status = 'optimal'
        difference = 0
        
        if current_val < ideal_min:
            status = 'low'
            difference = current_val - ideal_min
        elif current_val > ideal_max:
            status = 'high'
            difference = current_val - ideal_max
        
        analysis[param] = ParameterAnalysis(
            status=status, 
            current=current_val,
            ideal_min=float(ideal_min),
            ideal_max=float(ideal_max),
            difference=float(difference)
        )
    
    return analysis

@router.get("/available-models", summary="Get available models")
async def get_available_models():
    """Returns list of available prediction models"""
    return {
        "available_models": list(models.keys()),
        "available_crops": crop_classes,
        "features": feature_columns
    }

@router.post(
    "/check-suitability",
    response_model=SuitabilityResponse,
    summary="Check crop suitability",
    description="""Evaluates if a specific crop is suitable for given conditions.
    Returns suitability information and parameter analysis.""",
    tags=["Crop Analysis"]
)
async def check_crop_suitability(request: CropSuitabilityRequest):
    """
    Check suitability of specific crop for given parameters
    """
    try:
        # Check if system is initialized
        if not models or label_encoder is None:
            # Try to re-initialize
            try:
                initialize_system()
            except:
                raise HTTPException(
                    status_code=503,
                    detail="System not initialized. Please try again later."
                )
        
        # Prepare input data
        input_data = pd.DataFrame([[
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp, request.soil_moisture
        ]], columns=feature_columns)

        # Validate crop exists
        if request.crop not in crop_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found. Available crops: {', '.join(crop_classes)}"
            )
        
        # Get crop index
        crop_idx = label_encoder.transform([request.crop])[0]

        # Determine which models to use
        models_to_use = request.selected_models if request.selected_models else list(models.keys())
        valid_models = [m for m in models_to_use if m in models]
        
        if not valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"No valid models selected. Available models: {', '.join(models.keys())}"
            )

        # Get predictions from selected models
        model_confidence = {}
        confidences = []
        
        for model_name in valid_models:
            model_data = models[model_name]
            model = model_data['model']
            
            # Check if this model needs scaling
            requires_scaling = model_data.get('requires_scaling', False)
            
            if requires_scaling:
                scaler = model_data.get('scaler')
                if scaler is None:
                    # Fallback: use raw data if scaler not available
                    input_processed = input_data.values
                else:
                    input_processed = scaler.transform(input_data)
            else:
                input_processed = input_data.values
            
            # Get probability for the specific crop
            try:
                proba = model.predict_proba(input_processed)[0][crop_idx]
                model_confidence[model_name] = float(proba)
                confidences.append(proba)
            except Exception as e:
                # If predict_proba fails, use predict and binary confidence
                pred = model.predict(input_processed)[0]
                proba = 1.0 if pred == crop_idx else 0.0
                model_confidence[model_name] = float(proba)
                confidences.append(proba)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        is_suitable = avg_confidence >= 0.7  # 70% confidence threshold

        # Analyze parameters
        param_analysis = analyze_parameters(input_data, request.crop)

        return SuitabilityResponse(
            is_suitable=is_suitable,
            confidence=round(avg_confidence, 4),
            crop=request.crop,
            image_url=CROP_IMAGE_MAPPING.get(request.crop),
            parameters_analysis=param_analysis,
            model_used=valid_models,
            model_confidence=model_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/model-info/{model_name}", summary="Get model information")
async def get_model_info(model_name: str):
    """Returns detailed information about a specific model"""
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    model_data = models[model_name]
    
    # Load metrics file if available
    metrics_file = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    return {
        "model_name": model_data.get('model_name'),
        "model_type": model_data.get('model_type'),
        "requires_scaling": model_data.get('requires_scaling', False),
        "performance_metrics": metrics,
        "training_params": model_data.get('training_params', {}),
        "feature_importance": model_data.get('feature_importance', {}) if hasattr(model_data.get('model'), 'feature_importances_') else None
    }

# Optional: Add an endpoint to reload models (for development)
@router.post("/reload-models", summary="Reload all models (development only)")
async def reload_models():
    """Reload all models from disk (useful during development)"""
    try:
        initialize_system()
        return {"message": "Models reloaded successfully", "model_count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")