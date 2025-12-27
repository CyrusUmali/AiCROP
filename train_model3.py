import pandas as pd
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/enhanced_crop_data.csv"
MODEL_DIR = "models/binary"
RANDOM_SEED = 42

FEATURE_COLS = [
    'soil_ph',
    'fertility_ec',
    'humidity',
    'sunlight',
    'soil_temp',
    'soil_moisture'
]

TEST_SIZE = 0.3

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_crop_name(name: str) -> str:
    """Normalize crop names to a canonical form."""
    return (
        str(name)
        .strip()
        .replace('\xa0', ' ')
    )

def crop_to_filename(crop_name: str) -> str:
    """Convert crop name to model filename."""
    return (
        normalize_crop_name(crop_name)
        .lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('.', '')
        + "_binary_rf.pkl"
    )

# ==========================================================
# LOAD & CLEAN DATA
# ==========================================================
df = pd.read_csv(DATASET_PATH)

print(f"Loaded dataset: {len(df)} rows")

# Normalize labels ONCE (CRITICAL)
df['label'] = (
    df['label']
    .astype(str)
    .apply(normalize_crop_name)
)

print(f"Unique crops after normalization: {df['label'].nunique()}")

# ==========================================================
# TRAIN ONE BINARY MODEL PER CROP
# ==========================================================
summary = {}

for crop in sorted(df['label'].unique()):
    print(f"\nTraining binary suitability model for crop: {crop}")

    # Create binary target
    df_binary = df.copy()
    df_binary['target'] = (df_binary['label'] == crop).astype(int)

    X = df_binary[FEATURE_COLS]
    y = df_binary['target']

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # Random Forest Binary Classifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum())
    }

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1_score']:.4f}"
    )

    # Save model artifact
    artifact = {
        "crop": crop,
        "model": model,
        "feature_columns": FEATURE_COLS,
        "metrics": metrics,
        "random_seed": RANDOM_SEED
    }

    model_path = os.path.join(MODEL_DIR, crop_to_filename(crop))
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"✓ Saved model → {model_path}")

    summary[crop] = metrics

# ==========================================================
# SAVE TRAINING SUMMARY
# ==========================================================
summary_path = os.path.join(MODEL_DIR, "binary_training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("BINARY TRAINING COMPLETE")
print("="*80)
print(f"✓ Models saved in: {MODEL_DIR}")
print(f"✓ Summary saved to: {summary_path}")
