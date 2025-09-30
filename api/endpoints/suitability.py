from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, AsyncGenerator
import google.generativeai as genai
import os 
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter() 
genai.configure(api_key="AIzaSyCWiZmhjdh1GmYKnvJMLvgsY-bh20wYOZs") 

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
    
except FileNotFoundError as e:
    raise RuntimeError(f"Artifacts file not found: {e}")

# Language mapping
LANGUAGE_MAPPING = {
    "en": "English",
    "fil": "Filipino",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French"
    # Add more languages as needed
}

async def stream_gemini_suggestions(crop: str, deficiencies: dict, language: str = "en") -> AsyncGenerator[str, None]:
    """
    Stream actionable farming suggestions using Google's Gemini API
    
    Args:
        crop: Name of the crop
        deficiencies: Dictionary of deficient parameters and their values
        language: Language code for the response (default: 'en' for English)
    
    Yields:
        String chunks with formatted suggestions
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Get language name or default to English
        language_name = LANGUAGE_MAPPING.get(language, "English")
        
        prompt = f"""Act as an agricultural expert. Provide specific recommendations for growing {crop} given these conditions:
        
        Deficient Parameters:
        {", ".join(deficiencies.keys())}
        
        Current vs Ideal Values:
        { {param: f"{details.current} (ideal: {details.ideal_min}-{details.ideal_max})" 
           for param, details in deficiencies.items()}
        }

        Important Instructions:
        - Provide recommendations in {language_name} language
        - Use simple terms understandable by farmers
        - Include local measurement units where applicable
        - Consider regional agricultural practices

        Provide:
        1. Specific fertilizer recommendations with quantities
        2. Soil management techniques
        3. Environmental adjustments
        4. Any other relevant practices
        
        Formatting instructions:
        - Use simple bullet points with hyphens (-)
        - Keep each point concise
        - Avoid complex formatting
        - Use plain text only"""

        # Generate content with streaming
        response = await model.generate_content_async(prompt, stream=True)
        
        async for chunk in response:
            yield chunk.text

    except Exception as e:
        yield f"Error: {str(e)}"

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
    has_suggestions: bool = Field(
        False,
        description="Flag indicating if suggestions are available (can be requested separately)"
    )
    disclaimer: str = "AI suggestions should be verified with local agricultural experts"

class SuggestionRequest(BaseModel):
    """Request model for getting AI suggestions"""
    parameters: CropSuitabilityRequest
    deficient_params: List[str] = Field(
        ...,
        description="List of parameters that need improvement suggestions"
    )
    language: str = Field(
        "en",
        description="Language code for the suggestions (e.g., 'en' for English, 'fil' for Filipino)",
        example="fil"
    )

class SuggestionResponse(BaseModel):
    """Response model for AI suggestions"""
    suggestions: str
    disclaimer: str = "AI suggestions should be verified with local agricultural experts"

async def get_gemini_suggestions(crop: str, deficiencies: dict, language: str = "en") -> str:
    """
    Generate actionable farming suggestions using Google's Gemini API
    
    Args:
        crop: Name of the crop
        deficiencies: Dictionary of deficient parameters and their values
        language: Language code for the response
    
    Returns:
        String with formatted suggestions
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Get language name or default to English
        language_name = LANGUAGE_MAPPING.get(language, "English")
        
        prompt = f"""Act as an agricultural expert. Provide specific recommendations for growing {crop} given these conditions:
        
        Deficient Parameters:
        {", ".join(deficiencies.keys())}
        
        Current vs Ideal Values:
        { {param: f"{details.current} (ideal: {details.ideal_min}-{details.ideal_max})" 
           for param, details in deficiencies.items()}
        }

        Important Instructions:
        - Provide recommendations in {language_name} language
        - Use simple terms understandable by farmers
        - Include local measurement units where applicable
        - Consider regional agricultural practices

        Provide:
        1. Specific fertilizer recommendations with quantities
        2. Soil management techniques
        3. Environmental adjustments
        4. Any other relevant practices
        
        Formatting instructions:
        - Use simple bullet points with hyphens (-)
        - Keep each point concise
        - Avoid complex formatting
        - Use plain text only"""

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
    
    - **soil_ph**: Soil pH value
    - **fertility_ec**: Fertility/EC (Electrical Conductivity) in uS/cm
    - **humidity**: Relative humidity in %
    - **sunlight**:  lux
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

        # Validate crop exists
        try:
            crop_idx = le.transform([request.crop.lower()])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{request.crop}' not found in our database"
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
    "/get-suggestions-stream",
    summary="Stream AI-powered improvement suggestions",
    description="""Streams detailed improvement suggestions for crop cultivation
    based on parameter deficiencies using Google's Gemini AI.""",
    tags=["Crop Analysis"]
)
async def get_improvement_suggestions_stream(request: SuggestionRequest):
    """
    Stream AI-powered suggestions for improving crop cultivation conditions
    
    Requires the original parameters plus a list of deficient parameters
    that need improvement suggestions.
    
    Parameters:
    - parameters: Original crop parameters
    - deficient_params: List of parameters needing improvement
    - language: Language code for the response (e.g., 'en', 'fil')
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

        # Validate language
        if request.language not in LANGUAGE_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language code. Supported languages: {', '.join(LANGUAGE_MAPPING.keys())}"
            )

        # Prepare input data for parameter analysis
        input_data = pd.DataFrame([[
            request.parameters.soil_ph, request.parameters.fertility_ec, request.parameters.humidity,
            request.parameters.sunlight, request.parameters.soil_temp, request.parameters.soil_moisture
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
            return StreamingResponse(
                iter(["No significant deficiencies found in the selected parameters."]),
                media_type="text/plain"
            )

        # Return streaming response with language support
        return StreamingResponse(
            stream_gemini_suggestions(
                request.parameters.crop, 
                deficiencies,
                request.language
            ),
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )