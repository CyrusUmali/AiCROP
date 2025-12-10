from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from .crop_mapping import CROP_IMAGE_MAPPING
import json

router = APIRouter()

# Updated paths
MODELS_DIR = Path("models/individual")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables
models_data: Dict[str, dict] = {}  # Store model data dictionaries
label_encoder = None
feature_columns = None
crop_classes = None

def load_models_and_metadata():
    """Load all models and metadata"""
    global models_data, label_encoder, feature_columns, crop_classes
    
    # Load summary file for metadata
    with open(SUMMARY_PATH, 'r') as f:
        summary = json.load(f)
    
    feature_columns = summary['dataset_info']['features']
    crop_classes = summary['dataset_info']['crops']
    
    # Load all models
    models_data = {}
    for file_path in MODELS_DIR.glob("*.pkl"):
        if "_metrics" in file_path.name:
            continue
            
        with open(file_path, 'rb') as f:
            model_info = pickle.load(f)
            model_name = model_info.get('model_name', file_path.stem.replace('_', ' ').title())
            models_data[model_name] = model_info
    
    # Get label encoder from first model
    if models_data:
        first_model = next(iter(models_data.values()))
        label_encoder = first_model.get('label_encoder')
    
    print(f"✓ Prediction API initialized with {len(models_data)} models")

# Initialize on import
try:
    load_models_and_metadata()
except Exception as e:
    print(f"⚠ Warning: Failed to initialize prediction models: {e}")

class CropPredictionRequest(BaseModel):
    soil_ph: float
    fertility_ec: float
    humidity: float
    sunlight: float
    soil_temp: float
    soil_moisture: float

class PredictionResponse(BaseModel):
    crop: str
    confidence: float
    supporting_models: List[dict]
    image_url: Optional[str]
    warning: Optional[str] = None
    model_used: List[str]

ZERO_THRESHOLD = 1e-6


@router.get("/available-models")
async def get_available_prediction_models():
    """Get list of available models for prediction"""
    return {
        "available_models": list(models_data.keys()),
        "available_crops": crop_classes,
        "features": feature_columns
    }


@router.post("/predict")
async def predict_crop(
    request: CropPredictionRequest,
    min_confidence: float = 0.3,
    max_recommendations: int = 10,
    selected_models: Optional[List[str]] = None
):
    """Returns top N suitable crops with confidence scores and image links"""
    try:
        # Check if system is initialized
        if not models_data or label_encoder is None:
            try:
                load_models_and_metadata()
            except:
                raise HTTPException(
                    status_code=503,
                    detail="Prediction system not initialized"
                )
        
        # Prepare input
        input_data = pd.DataFrame([[ 
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp, request.soil_moisture
        ]], columns=feature_columns)

        # If no specific models are selected, use all models
        models_to_use = selected_models if selected_models else list(models_data.keys())
        
        # Validate selected models
        invalid_models = [m for m in models_to_use if m not in models_data]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid models selected: {invalid_models}. Available models: {list(models_data.keys())}"
            )

        # Determine if we should exclude 0-confidence predictions
        exclude_zero_conf = len(models_to_use) == 1

        # Get predictions from selected models
        all_predictions = []
        for model_name in models_to_use:
            model_info = models_data[model_name]
            model = model_info['model']
            requires_scaling = model_info.get('requires_scaling', False)
            
            # Prepare input based on model requirements
            if requires_scaling:
                scaler = model_info.get('scaler')
                if scaler:
                    input_processed = scaler.transform(input_data)
                else:
                    # If no scaler found, use raw data
                    input_processed = input_data.values
            else:
                input_processed = input_data.values
            
            try:
                # Get probability predictions
                probas = model.predict_proba(input_processed)[0]
                
                for class_idx, prob in enumerate(probas):
                    # Skip 0-confidence predictions ONLY if single model is selected
                    if exclude_zero_conf and prob <= ZERO_THRESHOLD:
                        continue
                    
                    crop_name = label_encoder.inverse_transform([class_idx])[0]
                    all_predictions.append({
                        "crop": crop_name,
                        "probability": float(prob),
                        "model": model_name
                    })
            except Exception as model_error:
                # Handle models that might not have predict_proba
                print(f"Warning: Model {model_name} failed: {model_error}")
                continue

        # If no predictions left after filtering
        if not all_predictions:
            return {
                "status": "error",
                "message": "No suitable crops found. Try lowering the confidence threshold or using different models."
            }

        # Aggregate probabilities by crop
        crop_scores = {}
        for pred in all_predictions:
            crop = pred["crop"]
            if crop not in crop_scores:
                crop_scores[crop] = {
                    "total_prob": 0,
                    "model_count": 0,
                    "models": [],
                    "model_used": set()  # Use set to avoid duplicates
                }
            
            crop_scores[crop]["total_prob"] += pred["probability"]
            crop_scores[crop]["model_count"] += 1
            crop_scores[crop]["models"].append({
                "model": pred["model"],
                "probability": pred["probability"]
            })
            crop_scores[crop]["model_used"].add(pred["model"])

        # Calculate average confidence and add image URL
        recommendations = []
        for crop, scores in crop_scores.items():
            avg_confidence = scores["total_prob"] / scores["model_count"]
            
            # Determine warning
            warning = None
            if avg_confidence < min_confidence:
                warning = "low_confidence"
            
            recommendations.append({
                "crop": crop,
                "confidence": round(avg_confidence, 4),
                "supporting_models": scores["models"],
                "image_url": CROP_IMAGE_MAPPING.get(crop, None),
                "warning": warning,
                "model_used": list(scores["model_used"])  # Convert set to list
            })

        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply confidence threshold (but always return at least top 3)
        suitable_crops = [
            crop for crop in recommendations 
            if crop["confidence"] >= min_confidence
        ]
        
        # Fallback to top 3 if threshold removes too many
        if len(suitable_crops) < 3:
            suitable_crops = recommendations[:3]
            # Add warnings for low-confidence fallbacks
            for crop in suitable_crops:
                if crop["confidence"] < min_confidence and crop.get("warning") is None:
                    crop["warning"] = "low_confidence_fallback"
        else:
            # Limit to max_recommendations
            suitable_crops = suitable_crops[:max_recommendations]

        return {
            "status": "success",
            "recommendations": suitable_crops,
            "confidence_threshold": min_confidence,
            "total_crops_considered": len(recommendations),
            "note": "Showing top 3 crops as fallback" if len(suitable_crops) < 3 else None,
            "excluded_zero_confidence": exclude_zero_conf,
            "models_used": models_to_use
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/best-model")
async def predict_with_best_model(
    request: CropPredictionRequest,
    min_confidence: float = 0.3,
    max_recommendations: int = 10
):
    """Predict using only the best performing model (based on accuracy)"""
    try:
        # Load model summary to find best model
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        best_model_name = summary.get('best_model')
        if not best_model_name:
            # If no best model specified, use the one with highest accuracy
            best_model_name = max(
                summary['model_performance'].items(),
                key=lambda x: x[1]['accuracy']
            )[0]
        
        # Call the main predict function with only the best model
        response = await predict_crop(
            request=request,
            min_confidence=min_confidence,
            max_recommendations=max_recommendations,
            selected_models=[best_model_name]
        )
        
        # Add best model info to response
        if response.get("status") == "success":
            response["best_model_used"] = best_model_name
            response["best_model_accuracy"] = summary['model_performance'][best_model_name]['accuracy']
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Best model prediction failed: {str(e)}"
        )


@router.post("/predict/ensemble")
async def predict_with_ensemble(
    request: CropPredictionRequest,
    ensemble_type: str = "all",  # "all", "tree_based", "top2"
    min_confidence: float = 0.3,
    max_recommendations: int = 10
):
    """Predict using different ensemble strategies"""
    try:
        # Load model summary for performance data
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        # Determine which models to use based on ensemble type
        all_models = list(models_data.keys())
        
        if ensemble_type == "tree_based":
            # Use only tree-based models
            models_to_use = [m for m in all_models if m in ['Random Forest', 'Decision Tree', 'XGBoost']]
        elif ensemble_type == "top2":
            # Use top 2 models by accuracy
            sorted_models = sorted(
                summary['model_performance'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )[:2]
            models_to_use = [m[0] for m in sorted_models]
        else:  # "all" or default
            models_to_use = all_models
        
        if not models_to_use:
            models_to_use = all_models  # Fallback to all models
        
        # Call the main predict function
        response = await predict_crop(
            request=request,
            min_confidence=min_confidence,
            max_recommendations=max_recommendations,
            selected_models=models_to_use
        )
        
        # Add ensemble info to response
        if response.get("status") == "success":
            response["ensemble_type"] = ensemble_type
            response["ensemble_models"] = models_to_use
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ensemble prediction failed: {str(e)}"
        )