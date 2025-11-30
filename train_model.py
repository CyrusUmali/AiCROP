import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
# dataset_path = 'dataset/crop_recommendation.csv'
dataset_path = 'dataset/enhanced_crop_data.csv'
df = pd.read_csv(dataset_path)

print(f"Dataset loaded: {len(df)} rows, {len(df['label'].unique())} unique crops")
print(f"Crops: {sorted(df['label'].unique())}\n")

# Check for missing values
if df.isnull().any().any():
    print("⚠ Warning: Dataset contains missing values")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"After removing missing values: {len(df)} rows\n")

# Create visualization directory
os.makedirs('visualizations', exist_ok=True)

# ============================================================================
# VISUALIZATION 1: Class Distribution
# ============================================================================
plt.figure(figsize=(14, 6))
crop_counts = df['label'].value_counts()
plt.bar(range(len(crop_counts)), crop_counts.values)
plt.xlabel('Crop')
plt.ylabel('Number of Samples')
plt.title('Dataset: Samples per Crop')
plt.xticks(range(len(crop_counts)), crop_counts.index, rotation=90)
plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/01_class_distribution.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Feature Distributions
# ============================================================================
feature_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {col.replace("_", " ").title()}')

plt.tight_layout()
plt.savefig('visualizations/02_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/02_feature_distributions.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Feature Correlation Heatmap
# ============================================================================
plt.figure(figsize=(10, 8))
correlation_matrix = df[feature_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/03_feature_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/03_feature_correlation.png")
plt.close()

# Encode the categorical label into numerical values
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Split data into features and target
X = df[feature_cols]
y = df['label_encoded']

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Target classes: {len(le.classes_)} crops\n")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

# Initialize StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=5, 
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
}

metrics = {}
confusion_matrices = {}

print("="*80)
print("TRAINING MODELS")
print("="*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        X_test_used = X_test_scaled
    else:
        model.fit(X_train, y_train)
        X_test_used = X_test

    # Predictions
    y_pred = model.predict(X_test_used)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    metrics[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=le.classes_, 
            zero_division=0
        ),
        'confusion_matrix': cm
    }
    
    print(f"✓ {name} trained - Accuracy: {metrics[name]['accuracy']:.4f}")

    # Save Decision Tree visualization
    if name == 'Decision Tree':
        dot_file_path = 'precomputation/decision_tree.dot'
        os.makedirs(os.path.dirname(dot_file_path), exist_ok=True)
        export_graphviz(
            model, 
            out_file=dot_file_path, 
            feature_names=X.columns.tolist(), 
            class_names=le.classes_, 
            filled=True, 
            rounded=True, 
            max_depth=3
        )
        print(f"  Decision tree visualization saved to {dot_file_path}")

# ============================================================================
# VISUALIZATION 4: Model Performance Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for idx, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    model_names = list(metrics.keys())
    values = [metrics[m][metric_key] for m in model_names]
    
    bars = ax.bar(model_names, values, alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} by Model')
    ax.set_ylim([0, 1.0])
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/04_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/04_model_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Confusion Matrices (Top 2 Models)
# ============================================================================
# Get top 2 models by accuracy
sorted_models = sorted(metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:2]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (model_name, model_metrics) in enumerate(sorted_models):
    cm = model_metrics['confusion_matrix']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=axes[idx], cbar_kws={"shrink": 0.8})
    
    axes[idx].set_title(f'{model_name}\nAccuracy: {model_metrics["accuracy"]:.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].tick_params(axis='both', which='major', labelsize=6)

plt.tight_layout()
plt.savefig('visualizations/05_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/05_confusion_matrices.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Feature Importance (Tree-based models)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

tree_models = ['Random Forest', 'XGBoost']
for idx, model_name in enumerate(tree_models):
    if model_name in models:
        model = models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[idx].bar(range(len(importances)), importances[indices], alpha=0.7, edgecolor='black')
            axes[idx].set_xticks(range(len(importances)))
            axes[idx].set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
            axes[idx].set_ylabel('Importance')
            axes[idx].set_title(f'Feature Importance - {model_name}')

plt.tight_layout()
plt.savefig('visualizations/06_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/06_feature_importance.png")
plt.close()

# Save all artifacts
artifacts = {
    'models': models,
    'metrics': metrics,
    'label_encoder': le,
    'scaler': scaler,
    'feature_columns': X.columns.tolist(),
    'crop_classes': le.classes_.tolist()
}

os.makedirs('precomputation', exist_ok=True)
with open('precomputation/training_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"✓ All artifacts saved to precomputation/training_artifacts.pkl")
print(f"✓ All visualizations saved to visualizations/")

# Print detailed results
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

for model_name, metric in metrics.items():
    print(f"\n{'='*80}")
    print(f"{model_name.upper()}")
    print(f"{'='*80}")
    print(f"Accuracy:  {metric['accuracy']:.4f}")
    print(f"Precision: {metric['precision']:.4f}")
    print(f"Recall:    {metric['recall']:.4f}")
    print(f"F1 Score:  {metric['f1_score']:.4f}")

# Summary table
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*80)
for model_name, metric in metrics.items():
    print(f"{model_name:<25} {metric['accuracy']:<12.4f} {metric['precision']:<12.4f} {metric['recall']:<12.4f} {metric['f1_score']:<12.4f}")

# Find best model
best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")