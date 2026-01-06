import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/enhanced_crop_data.csv"
MODEL_DIR = "models/binary"
RANDOM_SEED = 42
TEST_SIZE = 0.3
MIN_POSITIVE_SAMPLES = 20

FEATURE_COLS = [
    "soil_ph",
    "fertility_ec",
    "humidity",
    "sunlight",
    "soil_temp",
    "soil_moisture"
]

sns.set_style("whitegrid")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# HELPERS
# ==========================================================
def normalize_crop_name(name: str) -> str:
    return str(name).strip().replace("\xa0", " ")

def crop_to_filename(crop: str) -> str:
    return crop.lower().replace(" ", "_") + "_binary_rf.pkl"

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(DATASET_PATH)
df["label"] = df["label"].apply(normalize_crop_name)

# ==========================================================
# TRAIN BINARY MODELS
# ==========================================================
metrics_list = []

for crop in sorted(df["label"].unique()):
    df_bin = df.copy()
    df_bin["target"] = (df_bin["label"] == crop).astype(int)

    X = df_bin[FEATURE_COLS]
    y = df_bin["target"]

    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos < MIN_POSITIVE_SAMPLES:
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "crop": crop,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "positive_samples": pos,
        "negative_samples": neg
    }

    metrics_list.append(metrics)

    with open(os.path.join(MODEL_DIR, crop_to_filename(crop)), "wb") as f:
        pickle.dump({
            "crop": crop,
            "model": model,
            "features": FEATURE_COLS,
            "metrics": metrics
        }, f)

# ==========================================================
# SAVE METRICS DATA (FOR VISUALIZATION)
# ==========================================================
metrics_df = pd.DataFrame(metrics_list)

metrics_df.to_csv(
    os.path.join(MODEL_DIR, "binary_metrics_for_visualization.csv"),
    index=False
)

metrics_df.set_index("crop").to_json(
    os.path.join(MODEL_DIR, "binary_metrics_per_crop.json"),
    indent=2
)

# ==========================================================
# VISUALIZATION
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ROC-AUC vs PR-AUC
sns.boxplot(
    data=metrics_df[["roc_auc", "pr_auc"]],
    ax=axes[0, 0]
)
axes[0, 0].set_title("ROC-AUC & PR-AUC Distribution")

# Brier Score
sns.boxplot(
    data=metrics_df,
    y="brier_score",
    ax=axes[0, 1],
    color="salmon"
)
axes[0, 1].set_title("Brier Score (Lower is Better)")

# ROC vs Brier
sns.scatterplot(
    data=metrics_df,
    x="roc_auc",
    y="brier_score",
    size="positive_samples",
    hue="pr_auc",
    ax=axes[0, 2]
)
axes[0, 2].set_title("ROC-AUC vs Brier Score")

# Sample size vs ROC
sns.scatterplot(
    data=metrics_df,
    x="positive_samples",
    y="roc_auc",
    ax=axes[1, 0]
)
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("Sample Size vs ROC-AUC")

# Brier Histogram
sns.histplot(
    data=metrics_df,
    x="brier_score",
    bins=15,
    kde=True,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Brier Score Distribution")

# Correlation Heatmap
corr = metrics_df[
    ["roc_auc", "pr_auc", "brier_score", "positive_samples"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=axes[1, 2]
)
axes[1, 2].set_title("Metric Correlations")

plt.suptitle(
    "Binary Crop Suitability Model Diagnostics",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "binary_model_diagnostics.png"),
    dpi=300
)
plt.close()

print("✅ Binary training + visualization complete")
