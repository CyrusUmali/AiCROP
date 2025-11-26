from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional
import os 
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter() 

# Load all artifacts (including scaler)
ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")

try:
    # Load training artifacts
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    
    models = artifacts['models'] 
    metrics = artifacts['metrics']
    le = artifacts['label_encoder']
    scaler = artifacts['scaler']
    X_cols = artifacts['feature_columns']
    
    # Load dataset for reference (needed for parameter analysis)
    df = pd.read_csv(DATASET_PATH)
    
    # Print available crops for debugging
    # print("=== AVAILABLE CROPS ===")
    # print("Label Encoder classes:", list(le.classes_))
    # print("Dataset unique crops:", df['label'].unique() if 'label' in df.columns else "No 'label' column found")
    # print("=======================")
    
except FileNotFoundError as e:
    raise RuntimeError(f"Artifacts file not found: {e}")

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
    parameters_analysis: dict[str, ParameterAnalysis]
    model_used: List[str]
    disclaimer: str = "Results should be verified with local agricultural experts"

def analyze_parameters(input_data: pd.DataFrame, crop: str) -> dict[str, ParameterAnalysis]:
    """
    Compare input parameters with ideal ranges for the specified crop
    
    Args:
        input_data: DataFrame with input parameters
        crop: Name of the crop to analyze for
    
    Returns:
        Dictionary with parameter analysis results
    """
    crop_data = df[df['label'] == crop]  # No more .lower()
    if len(crop_data) == 0:
        return {}
    
    analysis = {}
    for param in X_cols:
        ideal_min = crop_data[param].quantile(0.25)
        ideal_max = crop_data[param].quantile(0.75)
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
    
    - **soil_ph**: Soil pH value
    - **fertility_ec**: Fertility/EC (Electrical Conductivity) in uS/cm
    - **humidity**: Relative humidity in %
    - **sunlight**: lux
    - **soil_temp**: Soil temperature in °C
    - **soil_moisture**: Soil moisture content in %
    - **crop**: Crop to check suitability for
    - **selected_models**: Optional list of models to use (default: all)
    """
    try:
        # Prepare input data
        input_data = pd.DataFrame([[
            request.soil_ph, request.fertility_ec, request.humidity,
            request.sunlight, request.soil_temp, request.soil_moisture
        ]], columns=X_cols)

        # Validate crop exists - NO MORE .lower()
        try:
            crop_idx = le.transform([request.crop])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found in our database. Available crops: {', '.join(le.classes_)}"
            )

        # Determine which models to use
        models_to_use = request.selected_models if request.selected_models else list(models.keys())
        valid_models = [m for m in models_to_use if m in models]
        
        if not valid_models:
            raise HTTPException(
                status_code=400,
                detail="No valid models selected. Available models: " + ", ".join(models.keys())
            )

        # Get predictions from selected models
        confidences = []
        for model_name in valid_models:
            model = models[model_name]
            # Use scaled data for Logistic Regression, raw for others
            if model_name == 'Logistic Regression':
                input_data_scaled = scaler.transform(input_data)
                proba = model.predict_proba(input_data_scaled)[0][crop_idx]
            else:
                proba = model.predict_proba(input_data)[0][crop_idx]
            confidences.append(proba)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        is_suitable = avg_confidence >= 0.7  # 70% confidence threshold

        # Analyze parameters - NO MORE .lower()
        param_analysis = analyze_parameters(input_data, request.crop)

        return SuitabilityResponse(
            is_suitable=is_suitable,
            confidence=round(avg_confidence, 4),
            crop=request.crop,
            image_url=CROP_IMAGE_MAPPING.get(request.crop),  # NO MORE .lower()
            parameters_analysis=param_analysis,
            model_used=valid_models
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )