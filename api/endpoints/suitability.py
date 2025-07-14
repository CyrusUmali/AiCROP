from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional
import google.generativeai as genai
import os 
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()
genai.configure(api_key="AIzaSyCWiZmhjdh1GmYKnvJMLvgsY-bh20wYOZs") 

# Model paths (adjust according to your project structure)
MODEL_DIR = Path("precomputation")
MODEL_PATH = MODEL_DIR / "models.pkl"
LE_PATH = MODEL_DIR / "label_encoder.pkl"
DATASET_PATH = Path("dataset/crop_recommendation.csv")

# Load models and encoder
try:
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    
    # Load dataset for reference
    df = pd.read_csv(DATASET_PATH)
    X_cols = df.drop('label', axis=1).columns.tolist()

except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found: {e}")

class CropSuitabilityRequest(BaseModel):
    """Request model for crop suitability check"""
    N: float = Field(..., description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., description="Phosphorus content in soil (kg/ha)")
    K: float = Field(..., description="Potassium content in soil (kg/ha)")
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    ph: float = Field(..., description="Soil pH value")
    rainfall: float = Field(..., description="Rainfall in mm")
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
    has_suggestions: bool = Field(
        False,
        description="Flag indicating if suggestions are available (can be requested separately)"
    )
    disclaimer: str = "AI suggestions should be verified with local agricultural experts"

class SuggestionRequest(BaseModel):
    """Request model for getting AI suggestions"""
    # Reuse the same parameters as CropSuitabilityRequest
    parameters: CropSuitabilityRequest
    deficient_params: List[str] = Field(
        ...,
        description="List of parameters that need improvement suggestions"
    )

class SuggestionResponse(BaseModel):
    """Response model for AI suggestions"""
    suggestions: str
    disclaimer: str = "AI suggestions should be verified with local agricultural experts"

async def get_gemini_suggestions(crop: str, deficiencies: dict) -> str:
    """
    Generate actionable farming suggestions using Google's Gemini API
    
    Args:
        crop: Name of the crop
        deficiencies: Dictionary of deficient parameters and their values
    
    Returns:
        String with formatted suggestions
    """
    try:
        # Set up the model
        # model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

        model = genai.GenerativeModel('gemini-1.5-flash')  # Fast & free

        #  model = genai.GenerativeModel('gemini-1.5-pro')  # or 'gemini-1.5-flash'
        
        # Build the prompt
        prompt = f"""Act as an agricultural expert. Provide specific recommendations for growing {crop} given these conditions:
        
        Deficient Parameters:
        {", ".join(deficiencies.keys())}
        
        Current vs Ideal Values:
        { {param: f"{details.current} (ideal: {details.ideal_min}-{details.ideal_max})" 
           for param, details in deficiencies.items()}
        }

        Provide:
        1. Specific fertilizer recommendations with quantities
        2. Soil management techniques
        3. Environmental adjustments
        4. Any other relevant practices
        
        Important formatting instructions:
        - Format as a clean bulleted list using simple hyphens (-) or bullet points (•)
        - Do not use asterisks (*) or other unnecessary decorative markings
        - Keep items concise and actionable
        - Use plain text suitable for farmers
        - Avoid any markdown or rich text formatting"""

        # Generate content
        response = await model.generate_content_async(prompt)
        return response.text

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


def analyze_parameters(input_data: pd.DataFrame, crop: str) -> dict[str, ParameterAnalysis]:
    """
    Compare input parameters with ideal ranges for the specified crop
    
    Args:
        input_data: DataFrame with input parameters
        crop: Name of the crop to analyze for
    
    Returns:
        Dictionary with parameter analysis results
    """
    crop_data = df[df['label'] == crop.lower()]
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
    Returns basic suitability information and parameter analysis.""",
    tags=["Crop Analysis"]
)
async def check_crop_suitability(request: CropSuitabilityRequest):
    """
    Check suitability of specific crop for given parameters
    
    - **N**: Nitrogen content in soil (kg/ha)
    - **P**: Phosphorus content in soil (kg/ha)
    - **K**: Potassium content in soil (kg/ha)
    - **temperature**: Temperature in °C
    - **humidity**: Relative humidity in %
    - **ph**: Soil pH value
    - **rainfall**: Rainfall in mm
    - **crop**: Crop to check suitability for
    - **selected_models**: Optional list of models to use (default: all)
    """
    try:
        # Prepare input data
        input_data = pd.DataFrame([[
            request.N, request.P, request.K,
            request.temperature, request.humidity,
            request.ph, request.rainfall
        ]], columns=X_cols)

        # Validate crop exists
        try:
            crop_idx = le.transform([request.crop.lower()])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found in our database"
            )

        # Determine which models to use
        models_to_use = request.selected_models if request.selected_models else models.keys()
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
            proba = model.predict_proba(input_data)[0][crop_idx]
            confidences.append(proba)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        is_suitable = avg_confidence >= 0.7  # 70% confidence threshold

        # Analyze parameters
        param_analysis = analyze_parameters(input_data, request.crop.lower())
        
        # Determine if there are any deficiencies that could have suggestions
        deficiencies = {
            param: details 
            for param, details in param_analysis.items() 
            if details.status != 'optimal'
        }
        has_suggestions = bool(deficiencies)

        return SuitabilityResponse(
            is_suitable=is_suitable,
            confidence=round(avg_confidence, 4),
            crop=request.crop,
            image_url=CROP_IMAGE_MAPPING.get(request.crop.lower()),
            parameters_analysis=param_analysis,
            model_used=valid_models,
            has_suggestions=has_suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post(
    "/get-suggestions",
    response_model=SuggestionResponse,
    summary="Get AI-powered improvement suggestions",
    description="""Generates detailed improvement suggestions for crop cultivation
    based on parameter deficiencies using Google's Gemini AI.""",
    tags=["Crop Analysis"]
)
async def get_improvement_suggestions(request: SuggestionRequest):
    """
    Get AI-powered suggestions for improving crop cultivation conditions
    
    Requires the original parameters plus a list of deficient parameters
    that need improvement suggestions.
    """
    try:
        # First validate the crop exists
        try:
            le.transform([request.parameters.crop.lower()])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.parameters.crop}' not found in our database"
            )

        # Prepare input data for parameter analysis
        input_data = pd.DataFrame([[
            request.parameters.N, request.parameters.P, request.parameters.K,
            request.parameters.temperature, request.parameters.humidity,
            request.parameters.ph, request.parameters.rainfall
        ]], columns=X_cols)

        # Get parameter analysis
        param_analysis = analyze_parameters(input_data, request.parameters.crop.lower())
        
        # Filter only the requested deficient parameters
        deficiencies = {
            param: details 
            for param, details in param_analysis.items() 
            if param in request.deficient_params and details.status != 'optimal'
        }

        if not deficiencies:
            return SuggestionResponse(
                suggestions="No significant deficiencies found in the selected parameters.",
            )

        # Get AI suggestions
        suggestions = await get_gemini_suggestions(request.parameters.crop, deficiencies)

        return SuggestionResponse(
            suggestions=suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )