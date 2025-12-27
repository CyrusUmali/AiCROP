import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
CSV_PATH = "dataset/SPC-soil-data(orig).csv"
MODEL_DIR = "models/binary"
OUTPUT_CSV = "results/spc_rowwise_suitability_results.csv"
DEBUG_LOG = "results/model_debug_log.txt"

FEATURE_COLS = [
    "soil_ph",
    "fertility_ec",
    "humidity",
    "sunlight",
    "soil_temp",
    "soil_moisture"
]

LABEL_COL = "label"
SUITABILITY_THRESHOLD = 0.6

os.makedirs("results", exist_ok=True)

# ==========================================================
# NORMALIZATION (SINGLE SOURCE OF TRUTH)
# ==========================================================
def normalize_crop_name(name):
    return (
        str(name)
        .strip()
        .replace("\xa0", " ")
        .replace("-", " ")
        .replace(".", "")
    )

def crop_to_filename(crop):
    return (
        normalize_crop_name(crop)
        .lower()
        .replace(" ", "_")
        + "_binary_rf.pkl"
    )

# ==========================================================
# MODEL CACHE
# ==========================================================
model_cache = {}

def load_crop_model(crop):
    crop = normalize_crop_name(crop)

    if crop in model_cache:
        return model_cache[crop]

    model_path = os.path.join(MODEL_DIR, crop_to_filename(crop))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    if "model" not in artifact:
        raise ValueError(f"Invalid artifact for crop: {crop}")

    model_cache[crop] = artifact
    return artifact

# ==========================================================
# CHECK MODEL AVAILABILITY (FIXED)
# ==========================================================
def check_all_models(csv_path):
    print("\n" + "=" * 60)
    print("DEBUG: CHECKING AVAILABLE MODELS")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    # 🔴 CRITICAL FIX: normalize labels BEFORE unique()
    df[LABEL_COL] = df[LABEL_COL].apply(normalize_crop_name)

    unique_crops = sorted(df[LABEL_COL].unique())

    available, missing = [], []

    for crop in unique_crops:
        model_path = os.path.join(MODEL_DIR, crop_to_filename(crop))
        if os.path.exists(model_path):
            available.append(crop)
        else:
            missing.append(crop)

    print(f"\nAvailable models: {len(available)}/{len(unique_crops)}")
    print(f"Missing models: {len(missing)}/{len(unique_crops)}")

    if missing:
        print("\nMissing crops:")
        for c in missing:
            print(f"  ✗ {c}")

    with open(DEBUG_LOG, "w") as f:
        f.write("MODEL CHECK\n")
        f.write("=" * 40 + "\n")
        f.write(f"Available ({len(available)}):\n")
        for c in available:
            f.write(f"{c}\n")
        f.write("\nMissing:\n")
        for c in missing:
            f.write(f"{c}\n")

    return available, missing

# ==========================================================
# ROW-WISE TESTING (FIXED)
# ==========================================================
def rowwise_test(csv_path):
    df = pd.read_csv(csv_path)
    df[LABEL_COL] = df[LABEL_COL].apply(normalize_crop_name)

    results = []

    for idx, row in df.iterrows():
        crop = row[LABEL_COL]

        try:
            artifact = load_crop_model(crop)
            model = artifact["model"]

            X = pd.DataFrame([row[FEATURE_COLS]], columns=FEATURE_COLS)
            prob = model.predict_proba(X)[0][1]

            results.append({
                **row.to_dict(),
                "suitability_probability": round(float(prob), 4),
                "suitable": prob >= SUITABILITY_THRESHOLD
            })

        except Exception as e:
            results.append({
                **row.to_dict(),
                "suitability_probability": None,
                "suitable": False,
                "error": str(e)
            })

    return pd.DataFrame(results)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    available, missing = check_all_models(CSV_PATH)

    print("\n" + "=" * 60)
    print("RUNNING ROW-WISE SUITABILITY TEST")
    print("=" * 60)

    result_df = rowwise_test(CSV_PATH)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDONE")
    print(f"Results saved to: {OUTPUT_CSV}")
