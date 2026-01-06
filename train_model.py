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
from typing import List, Dict, Any
from scipy.stats import rankdata

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
# Set this to False to skip visualization generation
CREATE_VISUALIZATIONS = True  # Change to False to skip visualizations

# Hyperparameter tuning configuration
# Set to False to use default parameters instead of RandomizedSearchCV
USE_HYPERPARAMETER_TUNING = False  # Change to False to skip hyperparameter tuning
RANDOM_SEARCH_ITERATIONS = 20  # Number of iterations for RandomizedSearchCV
RANDOM_SEARCH_CV = 3  # Cross-validation folds for RandomizedSearchCV
RANDOM_SEED = 42

# Top-K accuracy configuration
TOP_K_VALUES = [1, 3, 5]  # Values of K for Top-K accuracy

# ============================================================================

def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k_values: List[int] = None) -> Dict[int, float]:
    """
    Calculate Top-K accuracy for multiple K values.
    
    Args:
        y_true: True labels (shape: n_samples,) as numpy array
        y_proba: Predicted probabilities (shape: n_samples, n_classes)
        k_values: List of K values to compute
        
    Returns:
        Dictionary mapping K to accuracy score
    """
    if k_values is None:
        k_values = TOP_K_VALUES
    
    # Ensure y_true is numpy array (not pandas Series)
    y_true = np.array(y_true)
    
    n_classes = y_proba.shape[1]
    top_k_acc = {}
    
    # Get top K predictions for each sample
    for k in k_values:
        # Get indices of top K predictions
        top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
        
        # Check if true label is in top K predictions
        correct = np.array([y_true[i] in top_k_indices[i] for i in range(len(y_true))])
        
        # Calculate accuracy
        top_k_acc[k] = np.mean(correct)
    
    return top_k_acc

def mean_reciprocal_rank(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        y_true: True labels (shape: n_samples,) as numpy array
        y_proba: Predicted probabilities (shape: n_samples, n_classes)
        
    Returns:
        MRR score
    """
    # Ensure y_true is numpy array
    y_true = np.array(y_true)
    
    n_samples = len(y_true)
    reciprocal_ranks = []
    
    for i in range(n_samples):
        # Get ranking of true label (1-based ranking)
        ranking = rankdata(-y_proba[i], method='ordinal')  # Negative for descending order
        true_label_rank = ranking[y_true[i]]
        
        # Calculate reciprocal rank
        reciprocal_ranks.append(1.0 / true_label_rank)
    
    return np.mean(reciprocal_ranks)

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

# Convert y_train and y_test to numpy arrays for compatibility
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

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

    rf_tuner.fit(X_train, y_train_np)

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
        max_depth=10,
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

    xgb_tuner.fit(X_train, y_train_np)

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
top_k_metrics = {}  # Store Top-K metrics for each model
mrr_scores = {}     # Store MRR scores for each model

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train_np)
        X_test_used = X_test_scaled
        X_train_used = X_train_scaled
        # Save the scaler for logistic regression separately
        logistic_scaler = scaler
    else:
        model.fit(X_train, y_train_np)
        X_test_used = X_test
        X_train_used = X_train

    # Predictions
    y_pred = model.predict(X_test_used)
    
    # Get probability predictions for Top-K and MRR
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_used)
    else:
        # For models without predict_proba, use decision function or create dummy probabilities
        print(f"  ⚠ {name} doesn't have predict_proba method, skipping Top-K and MRR")
        y_proba = None
    
    # Calculate standard metrics
    cm = confusion_matrix(y_test_np, y_pred)
    confusion_matrices[name] = cm
    
    metrics[name] = {
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test_np, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test_np, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(
            y_test_np, y_pred, 
            target_names=le.classes_, 
            zero_division=0
        ),
        'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
    }
    
    # Calculate Top-K accuracy if probabilities are available
    if y_proba is not None:
        top_k_acc = top_k_accuracy(y_test_np, y_proba, TOP_K_VALUES)
        top_k_metrics[name] = top_k_acc
        
        # Calculate MRR
        mrr = mean_reciprocal_rank(y_test_np, y_proba)
        mrr_scores[name] = mrr
        
        # Add to metrics dictionary
        metrics[name]['top_k_accuracy'] = {f'top_{k}': acc for k, acc in top_k_acc.items()}
        metrics[name]['mrr'] = mrr
        
        print(f"✓ {name} trained - Accuracy: {metrics[name]['accuracy']:.4f}")
        print(f"  Top-K Accuracies: {', '.join([f'Top-{k}: {acc:.4f}' for k, acc in top_k_acc.items()])}")
        print(f"  MRR: {mrr:.4f}")
    else:
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
        if 'mrr' in json_metrics:
            json_metrics['mrr'] = float(json_metrics['mrr'])
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
# CREATE COMPARISON DATAFRAME FOR VISUALIZATION (if not already created)
# ============================================================================
if 'comparison_df' not in locals() and 'comparison_df' not in globals():
    # Create comparison data from metrics
    comparison_data = []
    
    for model_name, model_metrics in metrics.items():
        # Get metrics that were already calculated
        has_proba = hasattr(models[model_name], 'predict_proba')
        
        # Get Top-K metrics if available
        top_k_acc = model_metrics.get('top_k_accuracy', {})
        
        # Store comparison data
        model_info = {
            'Model': model_name,
            # Primary metrics for recommendation systems
            'MRR': model_metrics.get('mrr', 0) if has_proba else 0,
            'Top-1 Acc': top_k_acc.get('top_1', 0) if top_k_acc else 0,
            'Top-3 Acc': top_k_acc.get('top_3', 0) if top_k_acc else 0,
            'Top-5 Acc': top_k_acc.get('top_5', 0) if top_k_acc else 0,
            'Hit Rate@3': top_k_acc.get('top_3', 0) if top_k_acc else 0,  # Same as Top-3 Acc
            
            # Basic metrics
            'Accuracy': model_metrics['accuracy'],
            'F1-Score': model_metrics['f1_score'],
            
            # Model characteristics
            'Supports Ranking': has_proba,
            'Type': type(models[model_name]).__name__
        }
        
        comparison_data.append(model_info)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by Top-3 Accuracy (most relevant for recommendation)
    comparison_df = comparison_df.sort_values('Top-3 Acc', ascending=False)
    
    print("✓ Created comparison DataFrame for visualization")

# ============================================================================
# VISUALIZATION: Top-K Model Performance Comparison (Same format as Vis 4)
# ============================================================================
if CREATE_VISUALIZATIONS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define Top-K metrics to visualize (in the same 2x2 grid format)
    metric_names = ['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc', 'MRR']
    metric_labels = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Mean Reciprocal Rank (MRR)']
    
    for idx, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Get model names from comparison_df (already sorted by Top-3 Acc)
        model_names = comparison_df['Model'].tolist()
        values = comparison_df[metric_key].tolist()
        
        # Create bars (use consistent colors if you want)
        bars = ax.bar(model_names, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric_label} by Model')
        ax.set_ylim([0, 1.0])
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Top-K Model Performance Comparison for Crop Recommendation', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/08_topk_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: visualizations/08_topk_model_comparison.png")
    plt.close()



# ============================================================================
# VISUALIZATION 5: Top-K Accuracy and MRR - Improved Design
# ============================================================================
if CREATE_VISUALIZATIONS and any(model in top_k_metrics for model in models.keys()):
    # Create a focused visualization for Top-K
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================================
    # Plot 1: Top-K Accuracy Comparison (Grouped Bar Chart)
    # ============================================================
    ax1 = axes[0]
    
    # Prepare data for grouped bar chart
    models_with_topk = [name for name in models.keys() if name in top_k_metrics]
    if models_with_topk:
        x = np.arange(len(TOP_K_VALUES))
        width = 0.8 / len(models_with_topk)  # Width of each bar
        colors = plt.cm.Set2(np.linspace(0, 1, len(models_with_topk)))
        
        # Create grouped bars
        for idx, model_name in enumerate(models_with_topk):
            # Get accuracies for each K value
            accuracies = [top_k_metrics[model_name].get(k, 0) for k in TOP_K_VALUES]
            positions = x + idx * width - (len(models_with_topk) - 1) * width / 2
            
            bars = ax1.bar(positions, accuracies, width, 
                          label=model_name, color=colors[idx], 
                          edgecolor='black', alpha=0.8)
            
            # Add value labels on top of bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('K Value (Top-K)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Top-K Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Top-{k}' for k in TOP_K_VALUES], fontsize=11)
        ax1.set_ylim([0, 1.1])  # Add space for labels
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add explanation text
        explanation = "Top-K Accuracy: Percentage of test samples where\nthe correct crop is in the top K predictions"
        ax1.text(0.02, 0.98, explanation, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ============================================================
    # Plot 2: Top-K Accuracy Improvement Over Baseline
    # ============================================================
    ax2 = axes[1]
    
    if models_with_topk:
        # Calculate improvement from Top-1 to Top-3 and Top-5
        improvement_data = {}
        for model_name in models_with_topk:
            top1 = top_k_metrics[model_name].get(1, 0)
            top3 = top_k_metrics[model_name].get(3, 0)
            top5 = top_k_metrics[model_name].get(5, 0)
            
            improvement_data[model_name] = {
                'Top-1 to Top-3': top3 - top1,
                'Top-1 to Top-5': top5 - top1,
                'Top-3 to Top-5': top5 - top3
            }
        
        # Plot improvement as stacked or side-by-side bars
        x_positions = np.arange(len(improvement_data))
        metrics_to_show = ['Top-1 to Top-3', 'Top-1 to Top-5']
        colors_improvement = ['#2E86AB', '#A23B72']  # Different colors for improvements
        
        bottom_values = np.zeros(len(improvement_data))
        
        for metric_idx, metric in enumerate(metrics_to_show):
            values = [improvement_data[model][metric] for model in improvement_data.keys()]
            
            bars = ax2.bar(x_positions, values, bottom=bottom_values,
                          color=colors_improvement[metric_idx], edgecolor='black',
                          alpha=0.8, label=metric)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:  # Only show label if there's improvement
                    height = bar.get_y() + bar.get_height() / 2
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'+{value:.3f}', ha='center', va='center',
                            fontsize=9, fontweight='bold')
            
            bottom_values += values
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
        ax2.set_title('Improvement from Top-1 to Higher K Values', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(list(improvement_data.keys()), rotation=45, ha='right', fontsize=11)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Zero line
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.legend(loc='upper right', fontsize=10)
        
        # Add explanation
        explanation2 = "How much accuracy improves when considering\nmore crops in recommendations"
        ax2.text(0.02, 0.98, explanation2, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('visualizations/05_topk_improvement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/05_topk_improvement.png")
    
    # ============================================================
    # Additional Visualization: Top-K Performance Matrix
    # ============================================================
    if models_with_topk:
        fig2, ax3 = plt.subplots(figsize=(10, 8))
        
        # Create a matrix of Top-K accuracies
        matrix_data = []
        model_labels = []
        
        for model_name in models_with_topk:
            row = [top_k_metrics[model_name].get(k, 0) for k in TOP_K_VALUES]
            matrix_data.append(row)
            model_labels.append(model_name)
        
        matrix_data = np.array(matrix_data)
        
        # Create heatmap
        im = ax3.imshow(matrix_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(models_with_topk)):
            for j in range(len(TOP_K_VALUES)):
                text = ax3.text(j, i, f'{matrix_data[i, j]:.3f}',
                              ha='center', va='center', 
                              color='black' if matrix_data[i, j] < 0.7 else 'white',
                              fontweight='bold')
        
        # Set labels
        ax3.set_xticks(range(len(TOP_K_VALUES)))
        ax3.set_xticklabels([f'Top-{k}' for k in TOP_K_VALUES], fontsize=11)
        ax3.set_yticks(range(len(models_with_topk)))
        ax3.set_yticklabels(model_labels, fontsize=11)
        ax3.set_title('Top-K Accuracy Matrix\n(Darker = Better Performance)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Accuracy', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig('visualizations/05_topk_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/05_topk_matrix.png")
        plt.close(fig2)
    
    plt.close(fig)



# ============================================================================
# VISUALIZATION 6: Confusion Matrices (Top 2 Models)
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
    plt.savefig('visualizations/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/06_confusion_matrices.png")
    plt.close()

# ============================================================================
# VISUALIZATION 7: Feature Importance (Tree-based models)
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
    plt.savefig('visualizations/07_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/07_feature_importance.png")
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
    'training_date': str(pd.Timestamp.now().date()),
    'top_k_config': TOP_K_VALUES
}

for model_name, model_metrics in metrics.items():
    summary['model_performance'][model_name] = {
        'accuracy': float(model_metrics['accuracy']),
        'precision': float(model_metrics['precision']),
        'recall': float(model_metrics['recall']),
        'f1_score': float(model_metrics['f1_score'])
    }
    
    # Add Top-K and MRR if available
    if 'top_k_accuracy' in model_metrics:
        summary['model_performance'][model_name]['top_k_accuracy'] = {
            k: float(v) for k, v in model_metrics['top_k_accuracy'].items()
        }
    
    if 'mrr' in model_metrics:
        summary['model_performance'][model_name]['mrr'] = float(model_metrics['mrr'])

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"✓ Hyperparameter tuning: {'ENABLED' if USE_HYPERPARAMETER_TUNING else 'DISABLED'}")
print(f"✓ Top-K evaluation: K={TOP_K_VALUES}")
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
    
    if 'top_k_accuracy' in metric:
        print(f"Top-K Accuracy:")
        for k, acc in metric['top_k_accuracy'].items():
            print(f"  {k}: {acc:.4f}")
    
    if 'mrr' in metric:
        print(f"MRR:       {metric['mrr']:.4f}")

# Summary table
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Model':<25} {'Accuracy':<10} {'Top-3':<10} {'Top-5':<10} {'MRR':<10}")
print("-"*80)
for model_name, metric in metrics.items():
    top_3 = metric.get('top_k_accuracy', {}).get('top_3', 'N/A')
    top_5 = metric.get('top_k_accuracy', {}).get('top_5', 'N/A')
    mrr = metric.get('mrr', 'N/A')
    
    if isinstance(top_3, float):
        top_3 = f"{top_3:.4f}"
    if isinstance(top_5, float):
        top_5 = f"{top_5:.4f}"
    if isinstance(mrr, float):
        mrr = f"{mrr:.4f}"
    
    print(f"{model_name:<25} {metric['accuracy']:<10.4f} {top_3:<10} {top_5:<10} {mrr:<10}")

# Find best model by accuracy
best_model_by_acc = max(metrics.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Best Model by Accuracy: {best_model_by_acc[0]} (Accuracy: {best_model_by_acc[1]['accuracy']:.4f})")

# Find best model by MRR (if available)
models_with_mrr = [(name, metric['mrr']) for name, metric in metrics.items() if 'mrr' in metric]
if models_with_mrr:
    best_model_by_mrr = max(models_with_mrr, key=lambda x: x[1])
    print(f"🏆 Best Model by MRR: {best_model_by_mrr[0]} (MRR: {best_model_by_mrr[1]:.4f})")

print(f"\n💾 Individual model file: models/individual/{best_model_by_acc[0].lower().replace(' ', '_')}.pkl")
print(f"🔧 Hyperparameter tuning used: {USE_HYPERPARAMETER_TUNING}")