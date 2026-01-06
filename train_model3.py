import pandas as pd
import os
import pickle
import json

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/enhanced_crop_data.csv"
MODEL_DIR = "models/binary"
CONF_MATRIX_DIR = os.path.join(MODEL_DIR, "confusion_matrices")
METRICS_PLOT_DIR = os.path.join(MODEL_DIR, "metrics_plots")
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
os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(METRICS_PLOT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_crop_name(name: str) -> str:
    return str(name).strip().replace('\xa0', ' ')

def crop_to_filename(crop_name: str) -> str:
    return (
        normalize_crop_name(crop_name)
        .lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('.', '')
        + "_binary_rf.pkl"
    )

def plot_confusion_matrix_png(y_true, y_pred, crop, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Suitable", "Suitable"],
        yticklabels=["Not Suitable", "Suitable"]
    )
    plt.title(f"Confusion Matrix – {crop}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{crop_to_filename(crop).replace('.pkl','')}_cm.png")
    plt.savefig(png_path)
    plt.close()
    return png_path

def plot_metrics_box(metrics_df, save_path):
    """Boxplot showing distribution of key metrics across crops."""
    metric_cols = ["balanced_accuracy", "precision", "recall", "f1_score"]
    metrics_long = metrics_df.melt(id_vars="crop", value_vars=metric_cols,
                                   var_name="metric", value_name="value")
    plt.figure(figsize=(8,6))
    sns.boxplot(data=metrics_long, x="metric", y="value")
    plt.ylim(0,1)
    plt.title("Distribution of Binary Crop Suitability Metrics Across Crops")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(DATASET_PATH)
df['label'] = df['label'].astype(str).apply(normalize_crop_name)

print(f"Loaded dataset: {len(df)} rows")
print(f"Unique crops: {df['label'].nunique()}")

# ==========================================================
# TRAIN ONE BINARY MODEL PER CROP
# ==========================================================
summary = []

for crop in sorted(df['label'].unique()):
    # ... (training code remains the same, but WITHOUT print statements)
    
    df_binary = df.copy()
    df_binary['target'] = (df_binary['label'] == crop).astype(int)

    X = df_binary[FEATURE_COLS]
    y = df_binary['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    y_pred = model.predict(X_test)

    metrics = {
        "crop": crop,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    # Save confusion matrix PNG (silently)
    cm_path = plot_confusion_matrix_png(y_test, y_pred, crop, CONF_MATRIX_DIR)
    metrics["confusion_matrix_png"] = cm_path

    # Save model artifact (silently)
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

    summary.append(metrics)

# ==========================================================
# CREATE AND PRINT ONLY THE VISUALIZATION DATA
# ==========================================================
summary_df = pd.DataFrame(summary)

print("=" * 60)
print("DATA USED TO VISUALIZE THE CHART (summary_df):")
print("=" * 60)
print("\nDataFrame Structure:")
print(f"Rows: {len(summary_df)} (one per crop)")
print(f"Columns: {list(summary_df.columns)}")
print("\nFirst 5 rows of the data:")
print(summary_df.head())
print("\nStatistical Summary of Metrics:")
print(summary_df[['balanced_accuracy', 'precision', 'recall', 'f1_score']].describe())
print("=" * 60)

# The rest of the code for saving JSON and creating the plot continues...
summary_path = os.path.join(MODEL_DIR, "binary_training_summary.json")
summary_df.to_json(summary_path, orient="records", indent=2)

box_plot_path = os.path.join(METRICS_PLOT_DIR, "metrics_distribution.png")
plot_metrics_box(summary_df, box_plot_path)
print(f"✓ Boxplot saved → {box_plot_path}")