from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import json
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Paths
MODELS_DIR = Path("models/individual")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables
models: Dict[str, dict] = {}
label_encoder = None
feature_columns = None
crop_classes = None
df = None  # Dataset reference


# ------------------ Helper functions ------------------

def load_model(model_name: str) -> dict:
    model_filename = model_name.lower().replace(' ', '_') + '.pkl'
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_all_models() -> Dict[str, dict]:
    available_models = {}
    for file_path in MODELS_DIR.glob("*.pkl"):
        if "_metrics" in file_path.name:
            continue
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            model_name = model_data.get('model_name', file_path.stem)
            available_models[model_name] = model_data
    return available_models


def initialize_system():
    global models, label_encoder, feature_columns, crop_classes, df
    try:
        df = pd.read_csv(DATASET_PATH)

        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)

        crop_classes = summary['dataset_info']['crops']
        feature_columns = summary['dataset_info']['features']

        models = load_all_models()
        if not models:
            raise RuntimeError("No models found in models/individual directory")

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


# ------------------ Pydantic Models ------------------

class CropSuitabilityRequest(BaseModel):
    soil_ph: float = Field(..., description="Soil pH value")
    fertility_ec: float = Field(..., description="Fertility/EC (Electrical Conductivity) in dS/m")
    humidity: float = Field(..., description="Relative humidity in %")
    sunlight: float = Field(..., description="Sunlight intensity in hours/day or lux")
    soil_temp: float = Field(..., description="Soil temperature in °C")
    soil_moisture: float = Field(..., description="Soil moisture content in %")
    crop: str = Field(..., description="Crop to check suitability for")
    selected_models: Optional[List[str]] = Field(None, description="List of model names to use for prediction. If None, uses all available models.")


class ParameterAnalysis(BaseModel):
    status: str  # 'low', 'optimal', 'high'
    current: float
    ideal_min: float
    ideal_max: float
    difference: Optional[float] = None


class SuitabilityResponse(BaseModel):
    is_suitable: bool
    final_confidence: float
    parameter_confidence: float
    ml_confidence: Dict[str, float]
    crop: str
    image_url: Optional[str] = None
    parameters_analysis: Dict[str, ParameterAnalysis]
    model_used: List[str]
    disclaimer: str = "Results should be verified with local agricultural experts"
















# ------------------ Core Logic ------------------

def feature_score(value, min_val, max_val):
    if min_val <= value <= max_val:
        return 1.0
    range_width = max_val - min_val if max_val != min_val else 1
    if value < min_val:
        return max(0.0, 1 - abs(value - min_val) / range_width)
    else:
        return max(0.0, 1 - abs(value - max_val) / range_width)


def analyze_parameters(input_data: pd.DataFrame, crop: str) -> Dict[str, ParameterAnalysis]:
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


def calculate_parameter_confidence(input_data: pd.DataFrame, crop: str, feature_importance: List[float]) -> float:
    crop_data = df[df['label'] == crop]
    scores, weights = [], []
    for i, param in enumerate(feature_columns):
        min_v = crop_data[param].quantile(0.01)
        max_v = crop_data[param].quantile(0.99)
        val = input_data[param].values[0]
        s = feature_score(val, min_v, max_v)
        w = feature_importance[i] if i < len(feature_importance) else 1.0
        scores.append(s * w)
        weights.append(w)
    return sum(scores) / sum(weights) if weights else 0.0


# ------------------ API Endpoint ------------------




from typing import List, Dict, Optional, Tuple
import numpy as np

# ------------------ Pydantic Models ------------------

class AlternativeCrop(BaseModel):
    crop: str
    confidence: float
    reason: str
    image_url: Optional[str] = None
    parameter_mismatches: List[str] = Field(default_factory=list)

class SuitabilityResponse(BaseModel):
    is_suitable: bool
    final_confidence: float
    parameter_confidence: float
    ml_confidence: Dict[str, float]
    crop: str
    image_url: Optional[str] = None
    parameters_analysis: Dict[str, ParameterAnalysis]
    model_used: List[str]
    alternatives: List[AlternativeCrop] = Field(default_factory=list)
    disclaimer: str = "Results should be verified with local agricultural experts"


# ------------------ Helper Functions for Alternatives ------------------

def find_similar_crops(
    input_data: pd.DataFrame,
    target_crop: str,
    top_n: int = 5
) -> List[Tuple[str, float, List[str]]]:
    """Find similar crops based on parameter compatibility"""
    if df is None or len(feature_columns) == 0:
        return []
    
    similarities = []
    
    for crop in crop_classes:
        if crop == target_crop:
            continue
            
        crop_data = df[df['label'] == crop]
        if len(crop_data) == 0:
            continue
        
        # Calculate compatibility score
        scores = []
        mismatches = []
        
        for param in feature_columns:
            current_val = input_data[param].values[0]
            crop_min = crop_data[param].quantile(0.01)
            crop_max = crop_data[param].quantile(0.99)
            
            # Check if current value is within crop's preferred range
            if crop_min <= current_val <= crop_max:
                scores.append(1.0)
            else:
                # Calculate how far outside the range
                if current_val < crop_min:
                    diff = crop_min - current_val
                    range_width = crop_max - crop_min if crop_max != crop_min else 1
                    normalized_diff = diff / range_width
                    score = max(0.0, 1 - normalized_diff)
                else:
                    diff = current_val - crop_max
                    range_width = crop_max - crop_min if crop_max != crop_min else 1
                    normalized_diff = diff / range_width
                    score = max(0.0, 1 - normalized_diff)
                
                scores.append(score)
                
                # If score is below threshold, add to mismatches
                if score < 0.7:
                    if current_val < crop_min:
                        mismatches.append(f"{param} too low ({current_val:.1f} < {crop_min:.1f})")
                    else:
                        mismatches.append(f"{param} too high ({current_val:.1f} > {crop_max:.1f})")
        
        # Average score across all parameters
        avg_score = np.mean(scores) if scores else 0
        
        similarities.append((crop, avg_score, mismatches))
    
    # Sort by similarity score (descending) and take top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def generate_reason(
    crop: str,
    score: float,
    mismatches: List[str],
    parameter_analysis: Dict[str, ParameterAnalysis]
) -> str:
    """Generate a human-readable reason for recommendation"""
    if score >= 0.9:
        return f"Excellent match - conditions are nearly ideal for {crop}"
    elif score >= 0.8:
        return f"Very good match - conditions are well-suited for {crop}"
    elif score >= 0.7:
        return f"Good match - conditions are suitable for {crop}"
    elif score >= 0.6:
        reason = f"Moderate match - {crop} can grow with minor adjustments"
        if mismatches:
            reason += f" (check: {', '.join(mismatches[:2])})"
        return reason
    elif score >= 0.5:
        reason = f"Possible alternative - {crop} may require some adjustments"
        if mismatches:
            reason += f" (needs: {', '.join(mismatches[:3])})"
        return reason
    else:
        reason = f"Consider if other options fail - {crop} might struggle"
        if mismatches:
            reason += f" (issues: {', '.join(mismatches[:3])})"
        return reason




def find_best_alternatives(
    input_data: pd.DataFrame,
    target_crop: str,
    target_analysis: Dict[str, ParameterAnalysis],
    models_to_use: List[str],  # Add this parameter
    top_n: int = 6
) -> List[AlternativeCrop]:
    """Find and rank alternative crops using CONSISTENT scoring with main crop"""
    
    if df is None or not feature_columns:
        return []

    alternatives: List[AlternativeCrop] = []
    
    # Get average feature importance across selected models
    avg_feature_importance = [1.0] * len(feature_columns)
    if models_to_use:
        for param_idx in range(len(feature_columns)):
            weights = []
            for model_name in models_to_use:
                if model_name in models:
                    fi = models[model_name].get('feature_importance', [1.0]*len(feature_columns))
                    if len(fi) > param_idx:
                        weights.append(fi[param_idx])
            avg_feature_importance[param_idx] = np.mean(weights) if weights else 1.0
    
    # Evaluate each alternative crop
    for crop in crop_classes:
        if crop == target_crop:
            continue
            
        crop_data = df[df['label'] == crop]
        if len(crop_data) == 0:
            continue
        
        # ---------- Calculate ML Confidence (same as main crop) ----------
        ml_confidences = []
        for model_name in models_to_use:
            if model_name not in models:
                continue
                
            model_data = models[model_name]
            model = model_data['model']
            requires_scaling = model_data.get('requires_scaling', False)
            
            input_processed = input_data.values
            if requires_scaling:
                scaler = model_data.get('scaler')
                if scaler:
                    input_processed = scaler.transform(input_data)
            
            try:
                # Get probability for THIS specific alternative crop
                crop_idx = label_encoder.transform([crop])[0]
                proba = model.predict_proba(input_processed)[0][crop_idx]
            except Exception:
                # Fallback to hard prediction
                pred_idx = model.predict(input_processed)[0]
                proba = 1.0 if pred_idx == crop_idx else 0.0
            
            ml_confidences.append(proba)
        
        avg_ml_conf = np.mean(ml_confidences) if ml_confidences else 0
        
        # ---------- Calculate Parameter Confidence (same as main crop) ----------
        # Use SAME feature importance weights as main crop calculation
        param_conf = calculate_parameter_confidence(
            input_data, 
            crop, 
            avg_feature_importance  # Use average feature importance
        )
        
        # ---------- Calculate Final Confidence (same 80/20 weighting) ----------
        final_conf = 0.8 * param_conf + 0.2 * avg_ml_conf
        
        # ---------- Identify mismatches for the reason ----------
        mismatches = []
        for param in feature_columns:
            val = input_data[param].values[0]
            min_val = crop_data[param].quantile(0.01)
            max_val = crop_data[param].quantile(0.99)
            if val < min_val:
                mismatches.append(f"{param} too low ({val:.1f} < {min_val:.1f})")
            elif val > max_val:
                mismatches.append(f"{param} too high ({val:.1f} > {max_val:.1f})")
        
        # Generate reason based on final confidence
        reason = generate_reason(crop, final_conf, mismatches, target_analysis)
        
        alternative = AlternativeCrop(
            crop=crop,
            confidence=round(final_conf, 3),
            reason=reason,
            image_url=CROP_IMAGE_MAPPING.get(crop),
            parameter_mismatches=mismatches[:3]
        )
        alternatives.append(alternative)
    
    # Sort by final confidence (descending)
    alternatives.sort(key=lambda x: x.confidence, reverse=True)
    return alternatives[:top_n]



# ------------------ Updated API Endpoint ------------------

@router.post("/check-suitability", response_model=SuitabilityResponse, summary="Check crop suitability", tags=["Crop Analysis"])
async def check_crop_suitability(
    request: CropSuitabilityRequest,
    include_alternatives: bool = True,
    num_alternatives: int = 6
):
    try:
        if not models or label_encoder is None:
            try:
                initialize_system()
            except:
                raise HTTPException(status_code=503, detail="System not initialized. Please try again later.")

        input_data = pd.DataFrame([[
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp, request.soil_moisture
        ]], columns=feature_columns)

        if request.crop not in crop_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found. Available crops: {', '.join(crop_classes)}"
            )

        crop_idx = label_encoder.transform([request.crop])[0]

        models_to_use = request.selected_models if request.selected_models else list(models.keys())
        valid_models = [m for m in models_to_use if m in models]
        if not valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"No valid models selected. Available models: {', '.join(models.keys())}"
            )

        model_confidence = {}
        model_param_confidences = []

        for model_name in valid_models:
            model_data = models[model_name]
            model = model_data['model']
            requires_scaling = model_data.get('requires_scaling', False)

            if requires_scaling:
                scaler = model_data.get('scaler')
                input_processed = scaler.transform(input_data) if scaler else input_data.values
            else:
                input_processed = input_data.values

            # ML probability
            try:
                proba = model.predict_proba(input_processed)[0][crop_idx]
            except Exception:
                pred = model.predict(input_processed)[0]
                proba = 1.0 if pred == crop_idx else 0.0

            model_confidence[model_name] = float(proba)

            # Parameter confidence weighted by feature importance
            feature_importance = model_data.get('feature_importance', [1.0]*len(feature_columns))
            param_conf = calculate_parameter_confidence(input_data, request.crop, feature_importance)
            model_param_confidences.append(param_conf)

        avg_model_conf = sum(model_confidence.values()) / len(model_confidence) if model_confidence else 0
        avg_param_conf = sum(model_param_confidences) / len(model_param_confidences) if model_param_confidences else 0

        # Final confidence (tunable weights)
        final_confidence = 0.8 * avg_param_conf + 0.2 * avg_model_conf

        is_suitable = final_confidence >= 0.7  # Threshold

        param_analysis = analyze_parameters(input_data, request.crop)
        
        # Find alternative crops if requested
        alternatives = []
        if include_alternatives:
            alternatives = find_best_alternatives(
                input_data, 
                request.crop, 
                param_analysis,
                 models_to_use=valid_models, 
                top_n=num_alternatives
            )

        return SuitabilityResponse(
            is_suitable=is_suitable,
            final_confidence=round(final_confidence, 4),
            parameter_confidence=round(avg_param_conf, 4),
            ml_confidence=model_confidence,
            crop=request.crop,
            image_url=CROP_IMAGE_MAPPING.get(request.crop),
            parameters_analysis=param_analysis,
            model_used=valid_models,
            alternatives=alternatives
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



 

 