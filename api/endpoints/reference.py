from fastapi import APIRouter, Query
import pandas as pd
from pathlib import Path
import pickle
from .crop_mapping import CROP_IMAGE_MAPPING

router = APIRouter()

# Path to the dataset and models
DATASET_PATH = Path("dataset/crop_recommendation.csv")
MODEL_PATH = Path("precomputation/models.pkl")

# Load models from pickle file
with open(MODEL_PATH, "rb") as f:
    models = pickle.load(f)

# Get available models from the loaded models dictionary
AVAILABLE_MODELS = list(models.keys())

@router.get("/crop-requirements")
async def get_crop_requirements(model: str = Query(..., enum=AVAILABLE_MODELS)):
    try:
        # Validate the selected model
        if model not in AVAILABLE_MODELS:
            return {
                "status": "error",
                "message": f"Model '{model}' is not available. Choose from {AVAILABLE_MODELS}."
            }

        # Read the dataset
        df = pd.read_csv(DATASET_PATH)

        # Extract unique crops from the 'label' column
        crops = df["label"].unique()
        requirements = {}

        for crop in crops:
            # Filter the dataset for the current crop
            crop_df = df[df["label"] == crop]

            # Calculate min, max, and mean for each relevant column
            stats = crop_df.describe().loc[["min", "max", "mean"]].to_dict()

            # Add predictions from the selected model
            # Note: We need to ensure we're only using the features the model expects
            model_features = crop_df.drop('label', axis=1)

            # With this:
            mean_features = crop_df.drop('label', axis=1).mean().values.reshape(1, -1)
            prediction = int(models[model].predict(mean_features)[0])

            # Construct the crop requirement data with stats and image URL
            requirements[crop] = {
                "requirements": {
                    "N": {
                        "min": round(stats["N"]["min"], 2),
                        "max": round(stats["N"]["max"], 2),
                        "mean": round(stats["N"]["mean"], 2)
                    },
                    "P": {
                        "min": round(stats["P"]["min"], 2),
                        "max": round(stats["P"]["max"], 2),
                        "mean": round(stats["P"]["mean"], 2)
                    },
                    "K": {
                        "min": round(stats["K"]["min"], 2),
                        "max": round(stats["K"]["max"], 2),
                        "mean": round(stats["K"]["mean"], 2)
                    },
                    "temperature": {
                        "min": round(stats["temperature"]["min"], 2),
                        "max": round(stats["temperature"]["max"], 2),
                        "mean": round(stats["temperature"]["mean"], 2)
                    },
                    "humidity": {
                        "min": round(stats["humidity"]["min"], 2),
                        "max": round(stats["humidity"]["max"], 2),
                        "mean": round(stats["humidity"]["mean"], 2)
                    },
                    "ph": {
                        "min": round(stats["ph"]["min"], 2),
                        "max": round(stats["ph"]["max"], 2),
                        "mean": round(stats["ph"]["mean"], 2)
                    },
                    "rainfall": {
                        "min": round(stats["rainfall"]["min"], 2),
                        "max": round(stats["rainfall"]["max"], 2),
                        "mean": round(stats["rainfall"]["mean"], 2)
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
    except Exception as e:
        return {"status": "error", "message": str(e)}