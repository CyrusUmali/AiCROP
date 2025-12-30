from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Paths
MULTICLASS_MODELS_DIR = Path("models/individual")
BINARY_MODELS_DIR = Path("models/binary")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables
multiclass_models: Dict[str, dict] = {}
binary_models: Dict[str, dict] = {}
label_encoder = None
feature_columns = None
crop_classes = None
df = None

# Caches for performance
CROP_STATS_CACHE = {}

# Suitability threshold
SUITABILITY_THRESHOLD = 0.6

# Multiclass filtering threshold
MULTICLASS_FILTER_THRESHOLD = 0.1  # Minimum confidence to consider from multiclass models
MULTICLASS_FILTER_LIMIT = 15  # Maximum crops to consider from multiclass filtering

# ------------------ Normalization (Single Source of Truth) ------------------

def normalize_crop_name(name: str) -> str:
    """Normalize crop names consistently"""
    return (
        str(name)
        .strip()
        .replace("\xa0", " ")
        .replace("-", " ")
        .replace(".", "")
    )

def crop_to_filename(crop: str) -> str:
    """Convert crop name to binary model filename"""
    return (
        normalize_crop_name(crop)
        .lower()
        .replace(" ", "_")
        + "_binary_rf.pkl"
    )


# ------------------ Helper functions ------------------

def load_binary_models() -> Dict[str, dict]:
    """Load all binary crop suitability models with new structure"""
    models = {}
    
    if not BINARY_MODELS_DIR.exists():
        print(f"⚠ Binary models directory not found: {BINARY_MODELS_DIR}")
        return models
    
    for file_path in BINARY_MODELS_DIR.glob("*_binary_rf.pkl"):
        try:
            with open(file_path, 'rb') as f:
                artifact = pickle.load(f)
                
                # Extract crop name from filename
                crop_name = file_path.stem.replace('_binary_rf', '').replace('_', ' ').title()
                crop_name = normalize_crop_name(crop_name)
                
                # Validate artifact structure
                if 'model' not in artifact:
                    print(f"⚠ Invalid artifact for {crop_name}: missing 'model' key")
                    continue
                
                models[crop_name] = artifact
                
        except Exception as e:
            print(f"⚠ Failed to load {file_path}: {e}")
    
    return models


def initialize_system():
    global multiclass_models, binary_models, label_encoder, feature_columns, crop_classes, df
    
    try:
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Normalize crop names in dataset
        if 'label' in df.columns:
            df['label'] = df['label'].apply(normalize_crop_name)
        
        # Load summary
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        crop_classes = [normalize_crop_name(c) for c in summary['dataset_info']['crops']]
        feature_columns = summary['dataset_info']['features']
        
        # Load multiclass models
        multiclass_models = {}
        for file_path in MULTICLASS_MODELS_DIR.glob("*.pkl"):
            if "_metrics" in file_path.name:
                continue
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                model_name = model_data.get('model_name', file_path.stem)
                multiclass_models[model_name] = model_data
        
        # Load binary models with new structure
        binary_models = load_binary_models()
        
        # Get label encoder from first multiclass model
        if multiclass_models:
            first_model = next(iter(multiclass_models.values()))
            label_encoder = first_model.get('label_encoder')
        
        print(f"✓ System initialized")
        print(f"  - Multiclass models: {len(multiclass_models)}")
        print(f"  - Binary crop models: {len(binary_models)}")
        print(f"  - Available crops: {len(crop_classes)}")
        print(f"  - Binary model crops: {', '.join(list(binary_models.keys())[:5])}...")
        
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
    crop: Optional[str] = Field(None, description="Specific crop to check suitability for (optional)")
    use_binary_models: bool = Field(True, description="Use binary models for suitability analysis (more accurate)")
    use_multiclass_filter: bool = Field(True, description="Use multiclass models to filter crops before binary analysis")
    selected_models: Optional[List[str]] = Field(None, description="List of model names to use")
    multiclass_confidence_threshold: float = Field(0.05, description="Minimum confidence for crops from multiclass models")


class ParameterAnalysis(BaseModel):
    status: str  # 'low', 'optimal', 'high'
    current: float
    ideal_min: float
    ideal_max: float
    difference: Optional[float] = None


class AlternativeCrop(BaseModel):
    crop: str
    confidence: float
    reason: str
    image_url: Optional[str] = None
    parameter_mismatches: List[str] = Field(default_factory=list)


class EnhancedSuitabilityResponse(BaseModel):
    is_suitable: bool
    final_confidence: float
    parameter_confidence: float
    ml_confidence: Dict[str, float]
    crop: str
    image_url: Optional[str] = None
    parameters_analysis: Dict[str, ParameterAnalysis]
    model_used: List[str]
    model_type: str  # 'multiclass' or 'binary'
    alternatives: List[AlternativeCrop] = Field(default_factory=list)
    filtered_crops: Optional[List[str]] = Field(None, description="Crops filtered by multiclass model")
    disclaimer: str = "Results should be verified with local agricultural experts"


# ------------------ Core Logic ------------------

def feature_score(value, min_val, max_val):
    """Calculate parameter score with soft center-peaked approach."""
    
    # If value is exactly at the ideal center
    ideal_center = (min_val + max_val) / 2
    
    if min_val <= value <= max_val:
        # Within optimal range
        if value == ideal_center:
            return 1.0  # Perfect at center
        
        # Calculate distance from center (normalized 0-1)
        distance_from_center = abs(value - ideal_center) / ((max_val - min_val) / 2)
        
        # Soft scoring: 85-100% within range
        # Quadratic decay gives smooth gradient
        score = 1.0 - (distance_from_center ** 2) * 0.15
        
        return max(0.85, score)  # Minimum 85% within optimal range
    
    else:
        # Outside optimal range - linear decay
        range_width = max_val - min_val if max_val != min_val else 1
        
        if value < min_val:
            distance = (min_val - value) / range_width
        else:  # value > max_val
            distance = (value - max_val) / range_width
        
        # Exponential decay outside range for steeper penalty
        score = max(0.0, 0.85 * np.exp(-distance * 1.5))
        return score




def get_crop_stats(crop: str) -> dict:
    """Cache crop statistics to avoid repeated quantile calculations"""
    if crop not in CROP_STATS_CACHE:
        crop_data = df[df['label'] == crop]
        if len(crop_data) == 0:
            return None
        
        CROP_STATS_CACHE[crop] = {
            param: {
                'min': float(crop_data[param].quantile(0.01)),
                'max': float(crop_data[param].quantile(0.99))
            }
            for param in feature_columns
        }
    return CROP_STATS_CACHE[crop]


def analyze_parameters(input_data: pd.DataFrame, crop: str) -> Dict[str, ParameterAnalysis]:
    stats = get_crop_stats(crop)
    if not stats:
        return {}
    
    analysis = {}
    for param in feature_columns:
        ideal_min = stats[param]['min']
        ideal_max = stats[param]['max']
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
            ideal_min=ideal_min,
            ideal_max=ideal_max,
            difference=float(difference)
        )
    
    return analysis


def calculate_parameter_confidence(input_data: pd.DataFrame, crop: str, feature_importance: Optional[List[float]] = None) -> float:
    """Calculate confidence based on parameter ranges"""
    stats = get_crop_stats(crop)
    if not stats:
        return 0.0
    
    scores, weights = [], []
    
    for i, param in enumerate(feature_columns):
        min_v = stats[param]['min']
        max_v = stats[param]['max']
        val = input_data[param].values[0]
        s = feature_score(val, min_v, max_v)
        w = feature_importance[i] if feature_importance and i < len(feature_importance) else 1.0
        scores.append(s * w)
        weights.append(w)
    
    return sum(scores) / sum(weights) if weights else 0.0


# ------------------ Multiclass Model Functions ------------------

def filter_crops_with_multiclass(input_data: pd.DataFrame, 
                                 min_confidence: float = MULTICLASS_FILTER_THRESHOLD,
                                 max_crops: int = MULTICLASS_FILTER_LIMIT) -> List[str]:
    """
    Use multiclass models to filter down the list of crops to analyze.
    Returns a list of crop names that have reasonable confidence scores.
    """
    if not multiclass_models or not label_encoder:
        print("⚠ Multiclass models not available, using all binary models")
        return list(binary_models.keys())
    
    # Get predictions from all multiclass models
    crop_scores = {}
    
    for model_name, model_info in multiclass_models.items():
        model = model_info['model']
        requires_scaling = model_info.get('requires_scaling', False)
        
        # Prepare input based on model requirements
        if requires_scaling:
            scaler = model_info.get('scaler')
            if scaler:
                input_processed = scaler.transform(input_data)
            else:
                input_processed = input_data.values
        else:
            input_processed = input_data.values
        
        try:
            # Get probability predictions
            probas = model.predict_proba(input_processed)[0]
            
            for class_idx, prob in enumerate(probas):
                crop_name = label_encoder.inverse_transform([class_idx])[0]
                crop_name = normalize_crop_name(crop_name)
                
                if crop_name not in crop_scores:
                    crop_scores[crop_name] = {
                        "total_prob": 0,
                        "model_count": 0
                    }
                
                crop_scores[crop_name]["total_prob"] += prob
                crop_scores[crop_name]["model_count"] += 1
                
        except Exception as model_error:
            print(f"Warning: Multiclass model {model_name} failed: {model_error}")
            continue
    
    # Calculate average confidence for each crop
    filtered_crops = []
    for crop_name, scores in crop_scores.items():
        avg_confidence = scores["total_prob"] / scores["model_count"]
        
        # Only include crops above threshold and that have binary models
        if (avg_confidence >= min_confidence and 
            crop_name in binary_models):
            filtered_crops.append((crop_name, avg_confidence))
    
    # Sort by confidence (descending) and take top N
    filtered_crops.sort(key=lambda x: x[1], reverse=True)
    
    # Get just the crop names
    result = [crop for crop, _ in filtered_crops[:max_crops]]
    
    print(f"✓ Multiclass filtering: {len(result)} crops selected from {len(crop_scores)} total")
    return result


# ------------------ Binary Models Functions ------------------

def predict_with_binary_model(crop: str, input_data: pd.DataFrame) -> Dict[str, Any]:
    """Predict suitability using binary model for specific crop"""
    crop_normalized = normalize_crop_name(crop)
    
    if crop_normalized not in binary_models:
        raise ValueError(f"No binary model available for crop: {crop_normalized}")
    
    artifact = binary_models[crop_normalized]
    model = artifact['model']
    
    # Create DataFrame with correct column names
    X = pd.DataFrame([input_data.iloc[0].values], columns=feature_columns)
    
    # Get prediction probability (probability of suitable class = 1)
    proba = model.predict_proba(X)[0]
    suitability_score = float(proba[1])  # Probability of class 1 (suitable)
    
    # Get feature importance if available
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = {
            col: float(imp) 
            for col, imp in zip(feature_columns, model.feature_importances_)
        }
    
    return {
        'suitability_score': suitability_score,
        'prediction': int(suitability_score >= SUITABILITY_THRESHOLD),
        'feature_importance': feature_importance,
        'model_type': 'Random Forest Binary',
        'threshold': SUITABILITY_THRESHOLD
    }


def get_filtered_crop_suitabilities(input_data: pd.DataFrame, crops_to_check: List[str]) -> Dict[str, float]:
    """Get suitability scores for filtered crops using binary models"""
    scores = {}
    
    for crop in crops_to_check:
        try:
            result = predict_with_binary_model(crop, input_data)
            scores[crop] = result['suitability_score']
        except Exception as e:
            print(f"⚠ Error predicting for {crop}: {e}")
            scores[crop] = 0.0
    
    return scores


def generate_suitability_reason(crop: str, score: float) -> str:
    """Generate reason for suitability score"""
    if score >= 0.9:
        return f"Excellent match - ideal conditions for {crop}"
    elif score >= 0.8:
        return f"Very good conditions for {crop}"
    elif score >= 0.7:
        return f"Good conditions for {crop}"
    elif score >= SUITABILITY_THRESHOLD:
        return f"Moderate conditions - {crop} can grow with minor adjustments"
    elif score >= 0.5:
        return f"Borderline suitable - {crop} may require significant adjustments"
    elif score >= 0.3:
        return f"Poor conditions - {crop} will struggle to grow"
    else:
        return f"Very poor conditions - {crop} is not recommended"




def generate_alternative_with_confidence(
    crop: str, 
    input_data: pd.DataFrame,
    binary_result: dict
) -> AlternativeCrop:
    """Generate an AlternativeCrop with properly calculated final confidence"""
    # Calculate parameter analysis for this alternative crop
    param_analysis = analyze_parameters(input_data, crop)
    
    # Calculate parameter confidence
    feature_importance_list = list(binary_result['feature_importance'].values()) if binary_result['feature_importance'] else None
    param_conf = calculate_parameter_confidence(input_data, crop, feature_importance_list)
    
    # Binary model suitability score
    ml_score = binary_result['suitability_score']
    
    # Final confidence (weighted: 30% ML model, 70% parameter fit)
    final_confidence = 0.3 * ml_score + 0.7 * param_conf
    
    # Generate reason based on final confidence
    reason = generate_suitability_reason(crop, final_confidence)
    
    return AlternativeCrop(
        crop=crop,
        confidence=round(final_confidence, 4),
        reason=reason,
        image_url=CROP_IMAGE_MAPPING.get(crop),
        parameter_mismatches=[]
    )


@router.post("/check-suitability", 
             response_model=EnhancedSuitabilityResponse, 
             summary="Check crop suitability with binary models",
             tags=["Crop Analysis"])
async def check_crop_suitability_enhanced(
    request: CropSuitabilityRequest,
    include_alternatives: bool = False,
    num_alternatives: int = 6,
    min_confidence: float = 0.05,
    use_multiclass_filter: bool = True
):
    """
    Check crop suitability using binary models for more accurate analysis.
    Optionally get alternative crop recommendations filtered by multiclass model.
    """
    try:
        if not binary_models:
            initialize_system()
        
        # Normalize crop name
        crop_normalized = normalize_crop_name(request.crop)
        
        # Create input DataFrame
        input_data = pd.DataFrame([[
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp, request.soil_moisture
        ]], columns=feature_columns)
        
        # Check if crop exists in binary models
        if crop_normalized not in binary_models:
            available_crops = ', '.join(sorted(binary_models.keys())[:10])
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found in binary models. Available crops include: {available_crops}..."
            )
        
        # ===== Get suitability for requested crop =====
        binary_result = predict_with_binary_model(crop_normalized, input_data)
        suitability_score = binary_result['suitability_score']
        
        # Calculate parameter analysis for main crop
        param_analysis = analyze_parameters(input_data, crop_normalized)
        
        # Calculate parameter confidence for main crop
        feature_importance_list = list(binary_result['feature_importance'].values()) if binary_result['feature_importance'] else None
        param_conf = calculate_parameter_confidence(input_data, crop_normalized, feature_importance_list)
        
        # Final confidence for main crop (weighted: 30% ML model, 70% parameter fit)
        final_confidence = 0.3 * suitability_score + 0.7 * param_conf
        # is_suitable = suitability_score >= SUITABILITY_THRESHOLD

        # This is correct - using the final weighted confidence
        is_suitable = final_confidence >= SUITABILITY_THRESHOLD  # Uses 30% ML + 70% parameter confidence
        
        # ===== GET ALTERNATIVE CROPS =====
        alternatives = []
        if include_alternatives:
            candidate_crops = []
            
            # Method 1: Use multiclass model to filter candidates
            if use_multiclass_filter and multiclass_models:
                try:
                    candidate_crops = get_multiclass_candidates(
                        input_data=input_data,
                        min_confidence=min_confidence,
                        exclude_crop=crop_normalized,
                        max_candidates=num_alternatives * 3
                    )
                except Exception as e:
                    print(f"⚠ Multiclass filter failed: {e}")
                    # Fallback: use all binary models except requested crop
                    candidate_crops = [c for c in binary_models.keys() if c != crop_normalized]
            else:
                # Method 2: Use all binary models
                candidate_crops = [c for c in binary_models.keys() if c != crop_normalized]
            
            # Get binary results for candidate crops
            alternative_results = []
            for crop in candidate_crops:
                try:
                    alt_binary_result = predict_with_binary_model(crop, input_data)
                    
                    # Calculate parameter analysis for alternative
                    alt_param_analysis = analyze_parameters(input_data, crop)
                    
                    # Calculate parameter confidence for alternative
                    alt_feature_importance = list(alt_binary_result['feature_importance'].values()) if alt_binary_result['feature_importance'] else None
                    alt_param_conf = calculate_parameter_confidence(input_data, crop, alt_feature_importance)
                    
                    # Calculate final confidence for alternative (same weighted formula)
                    alt_final_confidence = 0.3 * alt_binary_result['suitability_score'] + 0.7 * alt_param_conf
                    
                    alternative_results.append({
                        'crop': crop,
                        'binary_result': alt_binary_result,
                        'param_analysis': alt_param_analysis,
                        'param_conf': alt_param_conf,
                        'final_confidence': alt_final_confidence
                    })
                except Exception as e:
                    print(f"⚠ Error processing alternative {crop}: {e}")
                    continue
            
            # Sort by final confidence (descending) and take top N
            alternative_results.sort(key=lambda x: x['final_confidence'], reverse=True)
            top_alternatives = alternative_results[:num_alternatives]
            
            # Create AlternativeCrop objects
            for alt in top_alternatives:
                reason = generate_suitability_reason(alt['crop'], alt['final_confidence'])
                
                # Extract parameter mismatches
                mismatches = []
                for param_name, analysis in alt['param_analysis'].items():
                    if analysis.status != 'optimal':
                        mismatches.append(f"{param_name} {analysis.status}")
                
                alternatives.append(AlternativeCrop(
                    crop=alt['crop'],
                    confidence=round(alt['final_confidence'], 4),
                    reason=reason,
                    image_url=CROP_IMAGE_MAPPING.get(alt['crop']),
                    parameter_mismatches=mismatches[:3]  # Limit to top 3 mismatches
                ))
        
        model_type = 'binary'
        model_used = [binary_result['model_type']]
        ml_confidence = {binary_result['model_type']: suitability_score}
        
        return EnhancedSuitabilityResponse(
            is_suitable=is_suitable,
            final_confidence=round(final_confidence, 4),
            parameter_confidence=round(param_conf, 4),
            ml_confidence=ml_confidence,
            crop=crop_normalized,
            image_url=CROP_IMAGE_MAPPING.get(crop_normalized),
            parameters_analysis=param_analysis,
            model_used=model_used,
            model_type=model_type,
            alternatives=alternatives,
            disclaimer="Results should be verified with local agricultural experts"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")






def get_multiclass_candidates(
    input_data: pd.DataFrame,
    min_confidence: float = 0.05,
    exclude_crop: Optional[str] = None,
    max_candidates: int = 20
) -> List[str]:
    """
    Get candidate crops from multiclass model predictions.
    Returns list of crop names sorted by confidence.
    """
    if not multiclass_models:
        return []
    
    # Prepare input for multiclass models
    input_array = input_data.values
    
    # Aggregate predictions from all multiclass models
    crop_predictions = {}
    
    for model_name, model_info in multiclass_models.items():
        model = model_info.get('model')
        label_encoder = model_info.get('label_encoder')
        requires_scaling = model_info.get('requires_scaling', False)
        
        if not model or not label_encoder:
            continue
        
        try:
            # Prepare input based on model requirements
            if requires_scaling:
                scaler = model_info.get('scaler')
                if scaler:
                    input_processed = scaler.transform(input_data)
                else:
                    input_processed = input_array
            else:
                input_processed = input_array
            
            # Get probability predictions
            probas = model.predict_proba(input_processed)[0]
            
            # Process each class
            for class_idx, prob in enumerate(probas):
                if prob < min_confidence:
                    continue
                    
                crop_name = label_encoder.inverse_transform([class_idx])[0]
                crop_name = normalize_crop_name(crop_name)
                
                # Skip excluded crop
                if exclude_crop and crop_name == exclude_crop:
                    continue
                
                # Aggregate probabilities
                if crop_name not in crop_predictions:
                    crop_predictions[crop_name] = {
                        'total_prob': 0,
                        'count': 0
                    }
                
                crop_predictions[crop_name]['total_prob'] += prob
                crop_predictions[crop_name]['count'] += 1
        
        except Exception as e:
            print(f"⚠ Multiclass model {model_name} failed: {e}")
            continue
    
    # Calculate average confidence for each crop
    candidates_with_conf = []
    for crop, scores in crop_predictions.items():
        avg_confidence = scores['total_prob'] / scores['count']
        candidates_with_conf.append((crop, avg_confidence))
    
    # Sort by confidence (descending) and limit
    candidates_with_conf.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the crop names
    return [crop for crop, _ in candidates_with_conf[:max_candidates]]


def get_filtered_binary_scores(
    input_data: pd.DataFrame,
    candidate_crops: List[str]
) -> Dict[str, float]:
    """
    Get binary suitability scores only for specific candidate crops.
    More efficient than getting all scores.
    """
    scores = {}
    
    for crop in candidate_crops:
        if crop in binary_models:
            try:
                result = predict_with_binary_model(crop, input_data)
                scores[crop] = result['suitability_score']
            except Exception as e:
                print(f"⚠ Error predicting for candidate {crop}: {e}")
                scores[crop] = 0.0
        else:
            # If crop doesn't have binary model, give it a low score
            scores[crop] = 0.0
    
    return scores