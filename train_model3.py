import pandas as pd
import os
import pickle
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/enhanced_crop_data.csv"
MODEL_DIR = "models/binary"
RANDOM_SEED = 42
TEST_SIZE = 0.3

FEATURE_COLS = [
    'soil_ph',
    'fertility_ec',
    'humidity',
    'sunlight',
    'soil_temp',
    'soil_moisture'
]

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_crop_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace('\xa0', ' ')
    )

def crop_to_filename(crop_name: str) -> str:
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

df['label'] = (
    df['label']
    .astype(str)
    .apply(normalize_crop_name)
)

unique_crops = sorted(df['label'].unique())
print(f"Unique crops after normalization: {len(unique_crops)}")

# ==========================================================
# TRAIN ONE BINARY MODEL PER CROP
# ==========================================================
summary = {}

for crop in unique_crops:
    print(f"\nTraining binary suitability model for crop: {crop}")

    # Binary target
    df_binary = df.copy()
    df_binary['target'] = (df_binary['label'] == crop).astype(int)

    X = df_binary[FEATURE_COLS]
    y = df_binary['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

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

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum())
    }

    print(
        f"Acc: {metrics['accuracy']:.3f} | "
        f"BalAcc: {metrics['balanced_accuracy']:.3f} | "
        f"Prec: {metrics['precision']:.3f} | "
        f"Rec: {metrics['recall']:.3f} | "
        f"F1: {metrics['f1_score']:.3f}"
    )

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

    summary[crop] = metrics

# ==========================================================
# SAVE METRICS SUMMARY
# ==========================================================
summary_path = os.path.join(MODEL_DIR, "binary_training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Per-crop metrics saved → {summary_path}")

# ==========================================================
# AGGREGATED EVALUATION (PAPER-READY)
# ==========================================================
summary_df = pd.DataFrame.from_dict(summary, orient="index")

print("\nAggregated binary model performance:")
print(summary_df.describe()[[
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score"
]])

metrics_to_plot = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score"
]

plt.figure(figsize=(11, 6))
plt.boxplot(
    [summary_df[m] for m in metrics_to_plot],
    labels=[m.replace("_", " ").title() for m in metrics_to_plot],
    showfliers=True
)

plt.ylabel("Score")
plt.title("Distribution of Binary Crop Suitability Model Performance (n = 35)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plot_path = os.path.join(
    MODEL_DIR,
    "binary_model_performance_distribution.png"
)

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✓ Aggregated performance plot saved → {plot_path}")

print("\n" + "=" * 80)
print("BINARY TRAINING + AGGREGATED EVALUATION COMPLETE")
print("=" * 80)
