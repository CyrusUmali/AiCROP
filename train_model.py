import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

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
import json

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
# Set this to False to skip visualization generation
CREATE_VISUALIZATIONS = True  # Change to False to skip visualizations

# Hyperparameter tuning configuration
# Set to False to use default parameters instead of RandomizedSearchCV
USE_HYPERPARAMETER_TUNING = True  # Change to False to skip hyperparameter tuning
RANDOM_SEARCH_ITERATIONS = 20  # Number of iterations for RandomizedSearchCV
RANDOM_SEARCH_CV = 3  # Cross-validation folds for RandomizedSearchCV
RANDOM_SEED = 42

# ============================================================================

# Set style (only if visualizations are enabled)
if CREATE_VISUALIZATIONS:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
# dataset_path = 'dataset/crop_recommendation.csv'
dataset_path = 'dataset/enhanced_crop_data.csv'
# dataset_path = 'dataset/augmented_crop_data.csv'
df = pd.read_csv(dataset_path)

print(f"Dataset loaded: {len(df)} rows, {len(df['label'].unique())} unique crops")
print(f"Crops: {sorted(df['label'].unique())}\n")

# Check for missing values
if df.isnull().any().any():
    print("⚠ Warning: Dataset contains missing values")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"After removing missing values: {len(df)} rows\n")

# Create visualization directory only if needed
if CREATE_VISUALIZATIONS:
    os.makedirs('visualizations', exist_ok=True)

# Create directories for saving individual models
os.makedirs('models', exist_ok=True)
os.makedirs('models/individual', exist_ok=True)

# ============================================================================
# VISUALIZATION 1: Class Distribution
# ============================================================================
if CREATE_VISUALIZATIONS:
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

if CREATE_VISUALIZATIONS:
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
if CREATE_VISUALIZATIONS:
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
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

# Initialize StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================
# RANDOM FOREST - WITH OR WITHOUT HYPERPARAMETER TUNING
# ==============================================================
if USE_HYPERPARAMETER_TUNING:
    print("="*80)
    print("PERFORMING HYPERPARAMETER TUNING FOR RANDOM FOREST")
    print("="*80)
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf_tuner = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        param_distributions=rf_param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=RANDOM_SEARCH_CV,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    rf_tuner.fit(X_train, y_train)

    print("\nBest Random Forest Parameters:")
    print(rf_tuner.best_params_)
    print(f"Best Cross-validation Accuracy: {rf_tuner.best_score_:.4f}")
    
    best_rf = rf_tuner.best_estimator_
else:
    print("="*80)
    print("USING DEFAULT PARAMETERS FOR RANDOM FOREST")
    print("="*80)
    best_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

# ==============================================================
# XGBOOST - WITH OR WITHOUT HYPERPARAMETER TUNING
# ==============================================================
if USE_HYPERPARAMETER_TUNING:
    print("\n" + "="*80)
    print("PERFORMING HYPERPARAMETER TUNING FOR XGBOOST")
    print("="*80)
    
    # Create XGBoost model with proper multi-class configuration
    xgb_model = XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob', 
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }

    # Use the model directly in RandomizedSearchCV
    xgb_tuner = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=RANDOM_SEARCH_CV,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    xgb_tuner.fit(X_train, y_train)

    print("\nBest XGBoost Parameters:")
    print(xgb_tuner.best_params_)
    print(f"Best Cross-validation Accuracy: {xgb_tuner.best_score_:.4f}")
    
    best_xgb = xgb_tuner.best_estimator_
else:
    print("\n" + "="*80)
    print("USING DEFAULT PARAMETERS FOR XGBOOST")
    print("="*80)
    best_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        eval_metric='mlogloss',
        objective='multi:softprob',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

# ==============================================================
# TRAIN ALL MODELS
# ==============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}

metrics = {}
confusion_matrices = {}

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        X_test_used = X_test_scaled
        # Save the scaler for logistic regression separately
        logistic_scaler = scaler
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
        'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
    }
    
    print(f"✓ {name} trained - Accuracy: {metrics[name]['accuracy']:.4f}")

    # ==============================================================
    # SAVE INDIVIDUAL MODEL
    # ==============================================================
    # Create a safe filename
    model_filename = name.lower().replace(' ', '_')
    
    # Prepare model artifacts
    model_artifacts = {
        'model': model,
        'model_name': name,
        'model_type': type(model).__name__,
        'feature_columns': X.columns.tolist(),
        'crop_classes': le.classes_.tolist(),
        'label_encoder': le,
        'performance_metrics': metrics[name],
        'training_params': model.get_params() if hasattr(model, 'get_params') else {},
        'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING if name in ['Random Forest', 'XGBoost'] else None,
        'random_seed': RANDOM_SEED
    }
    
    # Add scaler for logistic regression
    if name == 'Logistic Regression':
        model_artifacts['scaler'] = logistic_scaler
        model_artifacts['requires_scaling'] = True
    else:
        model_artifacts['requires_scaling'] = False
    
    # Save the model as pickle
    model_path = f'models/individual/{model_filename}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    # Save performance metrics as JSON for easy reading
    metrics_path = f'models/individual/{model_filename}_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = metrics[name].copy()
        json_metrics['accuracy'] = float(json_metrics['accuracy'])
        json_metrics['precision'] = float(json_metrics['precision'])
        json_metrics['recall'] = float(json_metrics['recall'])
        json_metrics['f1_score'] = float(json_metrics['f1_score'])
        json.dump(json_metrics, f, indent=2)
    
    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Metrics saved to: {metrics_path}")

    # Save Decision Tree visualization
    if name == 'Decision Tree' and CREATE_VISUALIZATIONS:
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
        print(f"  ✓ Decision tree visualization saved to {dot_file_path}")

# ============================================================================
# VISUALIZATION 4: Model Performance Comparison
# ============================================================================
if CREATE_VISUALIZATIONS:
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
if CREATE_VISUALIZATIONS:
    # Get top 2 models by accuracy
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (model_name, model_metrics) in enumerate(sorted_models):
        cm = np.array(model_metrics['confusion_matrix'])
        
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
if CREATE_VISUALIZATIONS:
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

# ============================================================================
# SAVE SUMMARY AND ALL MODELS TOGETHER
# ============================================================================
# Save all models together in one file
all_models_artifacts = {
    'models': models,
    'metrics': metrics,
    'label_encoder': le,
    'scaler': scaler,
    'feature_columns': X.columns.tolist(),
    'crop_classes': le.classes_.tolist(),
    'best_model': max(metrics.items(), key=lambda x: x[1]['accuracy'])[0],
    'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING,
    'random_seed': RANDOM_SEED,
    'training_timestamp': str(pd.Timestamp.now())
}

os.makedirs('precomputation', exist_ok=True)
with open('precomputation/all_models_artifacts.pkl', 'wb') as f:
    pickle.dump(all_models_artifacts, f)

# Save summary as JSON for easy reading
summary = {
    'dataset_info': {
        'total_samples': len(df),
        'num_crops': len(le.classes_),
        'crops': le.classes_.tolist(),
        'features': X.columns.tolist()
    },
    'model_performance': {},
    'best_model': max(metrics.items(), key=lambda x: x[1]['accuracy'])[0],
    'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING,
    'random_seed': RANDOM_SEED,
    'visualizations_created': CREATE_VISUALIZATIONS,
    'training_date': str(pd.Timestamp.now().date())
}

for model_name, model_metrics in metrics.items():
    summary['model_performance'][model_name] = {
        'accuracy': float(model_metrics['accuracy']),
        'precision': float(model_metrics['precision']),
        'recall': float(model_metrics['recall']),
        'f1_score': float(model_metrics['f1_score'])
    }

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"✓ Hyperparameter tuning: {'ENABLED' if USE_HYPERPARAMETER_TUNING else 'DISABLED'}")
print(f"✓ All models saved individually in models/individual/")
print(f"✓ All models together saved to precomputation/all_models_artifacts.pkl")
print(f"✓ Model summary saved to models/model_summary.json")

if CREATE_VISUALIZATIONS:
    print(f"✓ All visualizations saved to visualizations/")
    print(f"✓ Visualization mode: ON")
else:
    print(f"✓ Visualization mode: OFF (skipped visualization generation)")

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
print(f"   Individual model file: models/individual/{best_model[0].lower().replace(' ', '_')}.pkl")
print(f"   Hyperparameter tuning used: {USE_HYPERPARAMETER_TUNING}")