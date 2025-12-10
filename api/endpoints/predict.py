from fastapi import APIRouter
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Load all artifacts (including scaler)
ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f) 

models = artifacts['models']
metrics = artifacts['metrics']
le = artifacts['label_encoder']
scaler = artifacts['scaler']
X_cols = artifacts['feature_columns']

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
MODELS_REQUIRING_SCALING = ['Logistic Regression']  # Define which models need scaling


@router.post("/predict")
async def predict_crop(
    request: CropPredictionRequest,
    min_confidence: float = 0.3,
    max_recommendations: int = 10,
    selected_models: Optional[List[str]] = None
):
    """Returns top N suitable crops with confidence scores and image links"""
    try:
        # Prepare input
        input_data = pd.DataFrame([[ 
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp,
            request.soil_moisture
        ]], columns=X_cols)

        # Scale the input data for models that require it
        input_data_scaled = scaler.transform(input_data)

        # If no specific models are selected, use all models
        models_to_use = selected_models if selected_models else list(models.keys())

        # Determine if we should exclude 0-confidence predictions
        exclude_zero_conf = len(models_to_use) == 1

        # Get predictions from selected models
        all_predictions = []
        for model_name in models_to_use:
            model = models[model_name]
            
            # Use scaled data for Logistic Regression, raw data for others
            if model_name == 'Logistic Regression':
                probas = model.predict_proba(input_data_scaled)[0]
            else:
                probas = model.predict_proba(input_data)[0]
            
            for class_idx, prob in enumerate(probas):
                # Skip 0-confidence predictions ONLY if single model is selected
                if exclude_zero_conf and prob == ZERO_THRESHOLD:
                    continue
                crop_name = le.inverse_transform([class_idx])[0]
                all_predictions.append({
                    "crop": crop_name,
                    "probability": float(prob),
                    "model": model_name
                })

        # If no predictions left after filtering
        if not all_predictions:
            return {
                "status": "error",
                "message": "No suitable crops found (all predictions had 0 confidence)"
            }

        # Aggregate probabilities by crop
        crop_scores = {}
        for pred in all_predictions:
            if pred["crop"] not in crop_scores:
                crop_scores[pred["crop"]] = {
                    "total_prob": 0,
                    "model_count": 0,
                    "models": [],
                    "model_used": []
                }
            crop_scores[pred["crop"]]["total_prob"] += pred["probability"]
            crop_scores[pred["crop"]]["model_count"] += 1
            crop_scores[pred["crop"]]["models"].append({
                "model": pred["model"],
                "probability": pred["probability"]
            })
            crop_scores[pred["crop"]]["model_used"].append(pred["model"])

        # Calculate average confidence and add image URL
        recommendations = [{
            "crop": crop,
            "confidence": round(scores["total_prob"] / scores["model_count"], 4),
            "supporting_models": scores["models"],
            "image_url": CROP_IMAGE_MAPPING.get(crop, None),
            "warning": "low_confidence" if (scores["total_prob"] / scores["model_count"]) < min_confidence else None,
            "model_used": list(set(scores["model_used"]))  # Deduplicate model names
        } for crop, scores in crop_scores.items()]

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
            suitable_crops = suitable_crops[:max_recommendations]

        return {
            "status": "success",
            "recommendations": suitable_crops,
            "confidence_threshold": min_confidence,
            "total_crops_considered": len(recommendations),
            "note": "Showing top 3 crops as fallback" if len(suitable_crops) < 3 else None,
            "excluded_zero_confidence": exclude_zero_conf
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }