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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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

def plot_metrics_bar(metrics_df, save_path):
    """Plot metrics (precision, recall, F1, ROC-AUC, PR-AUC) per crop."""
    # Only keep numeric metric columns
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    metrics_numeric = metrics_df[["crop"] + metric_cols]

    metrics_long = metrics_numeric.melt(id_vars="crop", var_name="metric", value_name="value")
    
    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics_long, x="crop", y="value", hue="metric")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,1)
    plt.title("Per-Crop Binary Suitability Metrics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics_box(metrics_df, save_path):
    """Boxplot showing metric distribution across crops."""
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    metrics_numeric = metrics_df[["crop"] + metric_cols]

    metrics_long = metrics_numeric.melt(id_vars="crop", var_name="metric", value_name="value")
    
    plt.figure(figsize=(8,6))
    sns.boxplot(data=metrics_long, x="metric", y="value")
    plt.ylim(0,1)
    plt.title("Distribution of Metrics Across Crops")
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
    print(f"\nTraining binary suitability model for: {crop}")

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
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "crop": crop,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum())
    }

    # Save confusion matrix PNG
    cm_path = plot_confusion_matrix_png(y_test, y_pred, crop, CONF_MATRIX_DIR)
    metrics["confusion_matrix_png"] = cm_path

    print(
        f"Acc: {metrics['accuracy']:.3f} | "
        f"Prec: {metrics['precision']:.3f} | "
        f"Rec: {metrics['recall']:.3f} | "
        f"F1: {metrics['f1_score']:.3f} | "
        f"PR-AUC: {metrics['pr_auc']:.3f}"
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

    summary.append(metrics)

# ==========================================================
# SAVE SUMMARY JSON
# ==========================================================
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(MODEL_DIR, "binary_training_summary.json")
summary_df.to_json(summary_path, orient="records", indent=2)
print(f"\n✓ Summary saved → {summary_path}")

# ==========================================================
# AGGREGATED VISUALIZATIONS
# ==========================================================
bar_plot_path = os.path.join(METRICS_PLOT_DIR, "metrics_per_crop.png")
plot_metrics_bar(summary_df, bar_plot_path)
print(f"✓ Per-crop metrics bar plot saved → {bar_plot_path}")

box_plot_path = os.path.join(METRICS_PLOT_DIR, "metrics_distribution.png")
plot_metrics_box(summary_df, box_plot_path)
print(f"✓ Metrics distribution boxplot saved → {box_plot_path}")

print("\nBINARY SUITABILITY TRAINING COMPLETE")
