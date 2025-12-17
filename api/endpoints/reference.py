from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional
from pydantic import BaseModel  # ADD THIS IMPORT
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
    prediction: int
    image_url: Optional[str] = None
    ideal_range: Dict[str, Dict[str, float]]  # Added for better clarity

@router.get("/available-models")
async def get_requirements_models():
    """Get available models for crop requirements"""
    return {
        "available_models": AVAILABLE_MODELS,
        "available_crops": crop_classes,
        "features": feature_columns,
        "total_crops": len(crop_classes) if crop_classes else 0
    }

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
    - **minimal**: Just prediction and image URL
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
        
        # Get model data
        model_info = models_data[model]
        model_obj = model_info['model']
        requires_scaling = model_info.get('requires_scaling', False)
        scaler = model_info.get('scaler') if requires_scaling else None
        
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
            
            # Prepare features for prediction
            mean_features = crop_df[feature_columns].mean().values.reshape(1, -1)
            
            # Get prediction (handle scaling if needed)
            if requires_scaling and scaler is not None:
                mean_features_scaled = scaler.transform(mean_features)
                prediction = int(model_obj.predict(mean_features_scaled)[0])
            else:
                prediction = int(model_obj.predict(mean_features)[0])
            
            # Calculate statistics
            stats = {}
            ideal_range = {}
            
            for feature in feature_columns:
                feature_data = crop_df[feature]
                
                if format == "minimal":
                    # Only store min and max for minimal format
                    stats[feature] = RequirementStat(
                        min=round(float(feature_data.min()), 2),
                        max=round(float(feature_data.max()), 2),
                        mean=round(float(feature_data.mean()), 2),
                        unit=feature_units.get(feature, "")
                    )
                else:
                    # For detailed format, calculate percentiles for ideal range
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
                    "prediction": prediction,
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
                    "prediction": prediction,
                    "image_url": CROP_IMAGE_MAPPING.get(crop) if include_image else None,
                    "ideal_range": ideal_range
                }
            else:  # detailed format
                crop_data = {
                    "requirements": stats,
                    "prediction": prediction,
                    "image_url": CROP_IMAGE_MAPPING.get(crop) if include_image else None,
                    "ideal_range": ideal_range
                }
            
            requirements[crop] = crop_data
        
        return {
            "status": "success",
            "data": requirements,
            "model_used": model,
            "format": format
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/crop-requirements/{crop_name}")
async def get_specific_crop_requirements(
    crop_name: str,
    model: str = Query(None, description="Model to use for prediction. If None, uses best model."),
    format: str = Query("detailed", enum=["detailed", "summary", "minimal"])
):
    """
    Get ideal parameter ranges for a specific crop.
    
    If model is not specified, uses the best performing model.
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
        
        # Validate crop exists
        if crop_name not in df["label"].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Crop '{crop_name}' not found. Available crops: {', '.join(df['label'].unique())}"
            )
        
        # Determine which model to use
        if model is None:
            # Use the best model based on summary
            try:
                with open(SUMMARY_PATH, 'r') as f:
                    summary = json.load(f)
                model = summary.get('best_model', AVAILABLE_MODELS[0])
            except:
                model = AVAILABLE_MODELS[0]
        
        # Validate the selected model
        if model not in models_data:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not available"
            )
        
        # Get crop data
        crop_df = df[df["label"] == crop_name]
        
        if crop_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for crop '{crop_name}'"
            )
        
        # Get model data
        model_info = models_data[model]
        model_obj = model_info['model']
        requires_scaling = model_info.get('requires_scaling', False)
        scaler = model_info.get('scaler') if requires_scaling else None
        
        # Prepare features for prediction
        mean_features = crop_df[feature_columns].mean().values.reshape(1, -1)
        
        # Get prediction
        if requires_scaling and scaler is not None:
            mean_features_scaled = scaler.transform(mean_features)
            prediction = int(model_obj.predict(mean_features_scaled)[0])
        else:
            prediction = int(model_obj.predict(mean_features)[0])
        
        # Calculate statistics based on format
        stats = {}
        ideal_range = {}
        
        # Define units for each feature
        feature_units = {
            "soil_ph": "pH units",
            "fertility_ec": "dS/m",
            "humidity": "%",
            "sunlight": "hours/day",
            "soil_temp": "°C",
            "soil_moisture": "%"
        }
        
        for feature in feature_columns:
            feature_data = crop_df[feature]
            
            if format != "minimal":
                stats[feature] = {
                    "min": round(float(feature_data.min()), 2),
                    "max": round(float(feature_data.max()), 2),
                    "mean": round(float(feature_data.mean()), 2),
                    "unit": feature_units.get(feature, "")
                }
            
            # Calculate ideal range (25th to 75th percentile)
            ideal_min = round(float(feature_data.quantile(0.25)), 2)
            ideal_max = round(float(feature_data.quantile(0.75)), 2)
            
            ideal_range[feature] = {
                "ideal_min": ideal_min,
                "ideal_max": ideal_max,
                "optimal_mean": round(float(feature_data.mean()), 2),
                "unit": feature_units.get(feature, "")
            }
        
        # Construct response
        response = {
            "crop": crop_name,
            "model_used": model,
            "prediction": prediction,
            "image_url": CROP_IMAGE_MAPPING.get(crop_name),
            "ideal_range": ideal_range
        }
        
        if format != "minimal":
            response["requirements"] = stats
        
        return {
            "status": "success",
            "data": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/compare-crops")
async def compare_crops(
    crops: List[str] = Query(..., description="List of crops to compare"),
    feature: str = Query(None, description="Specific feature to compare. If None, compares all features"),
    format: str = Query("summary", enum=["detailed", "summary"])
):
    """
    Compare requirements between multiple crops.
    """
    try:
        # Check if system is initialized
        if df is None:
            raise HTTPException(
                status_code=503,
                detail="Requirements system not initialized"
            )
        
        # Check if dataset is empty
        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Dataset is empty"
            )
        
        # Validate crops
        available_crops = df["label"].unique()
        invalid_crops = [c for c in crops if c not in available_crops]
        if invalid_crops:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid crops: {invalid_crops}. Available crops: {', '.join(available_crops)}"
            )
        
        comparison = {}
        
        # If specific feature requested
        if feature:
            if feature not in feature_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{feature}' not found. Available features: {', '.join(feature_columns)}"
                )
            
            for crop in crops:
                crop_df = df[df["label"] == crop]
                if not crop_df.empty:
                    feature_data = crop_df[feature]
                    comparison[crop] = {
                        "min": round(float(feature_data.min()), 2),
                        "max": round(float(feature_data.max()), 2),
                        "mean": round(float(feature_data.mean()), 2),
                        "ideal_min": round(float(feature_data.quantile(0.25)), 2),
                        "ideal_max": round(float(feature_data.quantile(0.75)), 2)
                    }
        else:
            # Compare all features
            for crop in crops:
                crop_df = df[df["label"] == crop]
                if not crop_df.empty:
                    crop_stats = {}
                    for feat in feature_columns:
                        feat_data = crop_df[feat]
                        crop_stats[feat] = {
                            "min": round(float(feat_data.min()), 2),
                            "max": round(float(feat_data.max()), 2),
                            "mean": round(float(feat_data.mean()), 2),
                            "ideal_range": {
                                "min": round(float(feat_data.quantile(0.25)), 2),
                                "max": round(float(feat_data.quantile(0.75)), 2)
                            }
                        }
                    comparison[crop] = crop_stats
        
        return {
            "status": "success",
            "comparison": comparison,
            "crops_compared": crops,
            "feature": feature if feature else "all_features"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )