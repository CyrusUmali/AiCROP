from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from pathlib import Path
import pickle
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Path to the artifacts and dataset
ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")

try:
    # Load training artifacts (same as the other code)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    
    models = artifacts['models'] 
    metrics = artifacts['metrics']
    le = artifacts['label_encoder']
    scaler = artifacts['scaler']
    X_cols = artifacts['feature_columns']
    
    # Load dataset for reference
    df = pd.read_csv(DATASET_PATH)
    
except FileNotFoundError as e:
    raise RuntimeError(f"Artifacts file not found: {e}")

# Get available models from the loaded models dictionary
AVAILABLE_MODELS = list(models.keys())

@router.get("/crop-requirements")
async def get_crop_requirements(model: str = Query(..., enum=AVAILABLE_MODELS)):
    try:
        # Validate the selected model
        if model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not available. Choose from {AVAILABLE_MODELS}."
            )

        # Extract unique crops from the 'label' column
        crops = df["label"].unique()
        requirements = {}

        for crop in crops:
            # Filter the dataset for the current crop
            crop_df = df[df["label"] == crop]

            # Calculate min, max, and mean for each relevant column
            stats = crop_df.describe().loc[["min", "max", "mean"]].to_dict()

            # Prepare features for prediction (using the same X_cols as the other code)
            mean_features = crop_df[X_cols].mean().values.reshape(1, -1)
            
            # Get prediction (handle scaling for Logistic Regression like the other code)
            if model == 'Logistic Regression':
                mean_features_scaled = scaler.transform(mean_features)
                prediction = int(models[model].predict(mean_features_scaled)[0])
            else:
                prediction = int(models[model].predict(mean_features)[0])

            # Construct the crop requirement data with stats and image URL
            requirements[crop] = {
                "requirements": {
                    "soil_ph": {
                        "min": round(stats["soil_ph"]["min"], 2),
                        "max": round(stats["soil_ph"]["max"], 2),
                        "mean": round(stats["soil_ph"]["mean"], 2)
                    },
                    "fertility_ec": {
                        "min": round(stats["fertility_ec"]["min"], 2),
                        "max": round(stats["fertility_ec"]["max"], 2),
                        "mean": round(stats["fertility_ec"]["mean"], 2)
                    },
                    "humidity": {
                        "min": round(stats["humidity"]["min"], 2),
                        "max": round(stats["humidity"]["max"], 2),
                        "mean": round(stats["humidity"]["mean"], 2)
                    },
                    "sunlight": {
                        "min": round(stats["sunlight"]["min"], 2),
                        "max": round(stats["sunlight"]["max"], 2),
                        "mean": round(stats["sunlight"]["mean"], 2)
                    },
                    "soil_temp": {
                        "min": round(stats["soil_temp"]["min"], 2),
                        "max": round(stats["soil_temp"]["max"], 2),
                        "mean": round(stats["soil_temp"]["mean"], 2)
                    },
                    "soil_moisture": {
                        "min": round(stats["soil_moisture"]["min"], 2),
                        "max": round(stats["soil_moisture"]["max"], 2),
                        "mean": round(stats["soil_moisture"]["mean"], 2)
                    }
                },
                "prediction": prediction, 
                "image_url": CROP_IMAGE_MAPPING.get(crop.lower(), None)
            }

        return {
            "status": "success",
            "data": requirements,
            "model_used": model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )