from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional
from pydantic import BaseModel
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Updated paths
MODELS_DIR = Path("models/individual")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables
models_data: Dict[str, dict] = {}
df = None
feature_columns = None
crop_classes = None
label_encoder = None

def initialize_requirements_system():
    """Initialize the requirements system"""
    global models_data, df, feature_columns, crop_classes, label_encoder
    
    try:
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Load model summary for metadata
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
        
        print(f"✓ Requirements system initialized with {len(models_data)} models")
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize requirements system: {str(e)}")

# Initialize on import
try:
    initialize_requirements_system()
except Exception as e:
    print(f"⚠ Warning: Failed to initialize requirements system: {e}")

# Get available models
AVAILABLE_MODELS = list(models_data.keys()) if models_data else []

class RequirementStat(BaseModel):
    """Model for requirement statistics"""
    min: float
    max: float
    mean: float
    unit: str

class CropRequirement(BaseModel):
    """Model for crop requirement response"""
    requirements: Dict[str, RequirementStat]
    image_url: Optional[str] = None
    ideal_range: Dict[str, Dict[str, float]]  # Added for better clarity

@router.get("/crop-requirements")
async def get_crop_requirements(
    model: str = Query(..., description="Model to use for prediction"),
    include_image: bool = Query(True, description="Include image URLs in response"),
    format: str = Query("detailed", enum=["detailed", "summary", "minimal"])
):
    """
    Get ideal parameter ranges for all crops using a specific model.
    
    - **detailed**: Full statistics with min, max, mean
    - **summary**: Only min and max ranges
    - **minimal**: Just ideal range and image URL
    """
    try:
        # Check if system is initialized
        if not models_data or df is None:
            try:
                initialize_requirements_system()
            except:
                raise HTTPException(
                    status_code=503,
                    detail="Requirements system not initialized"
                )
        
        # Validate the selected model
        if model not in models_data:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not available. Choose from: {', '.join(AVAILABLE_MODELS)}"
            )
        
        # Get model data (still needed for any feature processing)
        model_info = models_data[model]
        
        # Extract unique crops
        if 'label' not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Dataset does not contain 'label' column"
            )
        
        crops = df["label"].unique()
        requirements = {}
        
        # Define units for each feature
        feature_units = {
            "soil_ph": "pH units",
            "fertility_ec": "dS/m",
            "humidity": "%",
            "sunlight": "hours/day",
            "soil_temp": "°C",
            "soil_moisture": "%"
        }
        
        for crop in crops:
            # Filter the dataset for the current crop
            crop_df = df[df["label"] == crop]
            
            if crop_df.empty:
                continue
            
            # Calculate statistics and ideal ranges
            stats = {}
            ideal_range = {}
            
            for feature in feature_columns:
                feature_data = crop_df[feature]
                
                if format == "minimal":
                    # Only store ideal range for minimal format
                    ideal_min = round(float(feature_data.quantile(0.25)), 2)  # 25th percentile
                    ideal_max = round(float(feature_data.quantile(0.75)), 2)  # 75th percentile
                    
                    ideal_range[feature] = {
                        "ideal_min": ideal_min,
                        "ideal_max": ideal_max,
                        "optimal_mean": round(float(feature_data.mean()), 2)
                    }
                else: 
                    # For detailed/summary format, calculate statistics
                    ideal_min = round(float(feature_data.quantile(0.25)), 2)  # 25th percentile
                    ideal_max = round(float(feature_data.quantile(0.75)), 2)  # 75th percentile
                    
                    stats[feature] = RequirementStat(
                        min=round(float(feature_data.min()), 2),
                        max=round(float(feature_data.max()), 2),
                        mean=round(float(feature_data.mean()), 2),
                        unit=feature_units.get(feature, "")
                    )
                    
                    ideal_range[feature] = {
                        "ideal_min": ideal_min,
                        "ideal_max": ideal_max,
                        "optimal_mean": round(float(feature_data.mean()), 2)
                    }
            
            # Create crop requirement data based on format
            if format == "minimal":
                crop_data = {
                    "requirements": {},  # Empty for minimal format
                    "image_url": CROP_IMAGE_MAPPING.get(crop) if include_image else None,
                    "ideal_range": ideal_range
                }
            elif format == "summary":
                # Only include min and max in summary format
                summary_stats = {}
                for feature in feature_columns:
                    feature_data = crop_df[feature]
                    summary_stats[feature] = {
                        "min": round(float(feature_data.min()), 2),
                        "max": round(float(feature_data.max()), 2),
                        "mean": round(float(feature_data.mean()), 2),
                        "unit": feature_units.get(feature, "")
                    }
                
                crop_data = {
                    "requirements": summary_stats,
                    "image_url": CROP_IMAGE_MAPPING.get(crop) if include_image else None,
                    "ideal_range": ideal_range
                }
            else:  # detailed format
                crop_data = {
                    "requirements": stats,
                    "image_url": CROP_IMAGE_MAPPING.get(crop) if include_image else None,
                    "ideal_range": ideal_range
                }
            
            requirements[crop] = crop_data
        
        return {
            "status": "success",
            "data": requirements,
            "model_used": model,
            "format": format,
            "total_crops": len(requirements)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )