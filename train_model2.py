import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ndcg_score
)
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import io

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
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

 
# ============================================================================
# ENHANCED: RANKING METRICS FOR RECOMMENDATION SYSTEMS
# ============================================================================


# ============================================================================
# ENHANCED: RANKING METRICS FOR RECOMMENDATION SYSTEMS (FIXED)
# ============================================================================

def evaluate_ranking_metrics(y_true, y_pred_proba, label_encoder, k_values=[1, 3, 5, 10]):
    """
    Evaluate ranking metrics for recommendation systems.
    
    Args:
        y_true: True labels (encoded) - can be pandas Series or numpy array
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        label_encoder: LabelEncoder for decoding classes
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary with ranking metrics
    """
    n_samples = len(y_true)
    n_classes = len(label_encoder.classes_)
    
    results = {}
    
    # Initialize metrics accumulators
    mrr_sum = 0.0
    precision_at_k = {k: 0.0 for k in k_values}
    recall_at_k = {k: 0.0 for k in k_values}
    map_sum = 0.0  # Mean Average Precision
    hit_at_k = {k: 0.0 for k in k_values}
    
    # Convert y_true to numpy array to handle both pandas Series and numpy arrays
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    
    
    # For each sample
    for i in range(n_samples):
        true_label = y_true_np[i]
        probabilities = y_pred_proba[i]
        
        # Get ranked list of predictions (descending probability)
        ranked_indices = np.argsort(probabilities)[::-1]
        
        # Calculate reciprocal rank (for MRR)
        rank_position = np.where(ranked_indices == true_label)[0]
        if len(rank_position) > 0:
            mrr_sum += 1.0 / (rank_position[0] + 1)
        else:
            # If true label not found in predictions (shouldn't happen)
            mrr_sum += 0.0
        
        # Calculate Average Precision (AP)
        relevant_ranks = []
        for k in range(1, n_classes + 1):
            if ranked_indices[k-1] == true_label:
                relevant_ranks.append(k)
        
        # Calculate precision at each relevant rank
        precisions_at_rel = []
        for rank in relevant_ranks:
            precisions_at_rel.append(1.0 / rank)
        
        # Average Precision for this query
        ap = np.mean(precisions_at_rel) if precisions_at_rel else 0.0
        map_sum += ap
        
        # Calculate Precision@K, Recall@K, and Hit@K
        for k in k_values:
            if k <= n_classes:
                top_k = ranked_indices[:k]
                
                # Hit@K (1 if true label is in top K)
                hit_at_k[k] += 1 if true_label in top_k else 0
                
                # Precision@K (proportion of relevant in top K)
                # For single relevant item, precision is either 1/k or 0
                precision = 1.0 / k if true_label in top_k else 0.0
                precision_at_k[k] += precision
                
                # Recall@K (1 if found, 0 otherwise for single relevant item)
                recall = 1.0 if true_label in top_k else 0.0
                recall_at_k[k] += recall
    
    # Calculate final metrics
    results['mrr'] = mrr_sum / n_samples if n_samples > 0 else 0.0
    results['map'] = map_sum / n_samples if n_samples > 0 else 0.0
    
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k[k] / n_samples if n_samples > 0 else 0.0
        results[f'recall@{k}'] = recall_at_k[k] / n_samples if n_samples > 0 else 0.0
        results[f'hit_rate@{k}'] = hit_at_k[k] / n_samples if n_samples > 0 else 0.0
    
    return results

def evaluate_top_n_performance(y_true, y_pred_proba, label_encoder, top_n_values=[1, 3, 5, 10]):
    """
    Evaluate model performance for top-N recommendations.
    This matches how the endpoint will be used.
    
    Args:
        y_true: True labels (encoded)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        label_encoder: LabelEncoder for decoding classes
        top_n_values: List of N values to evaluate (e.g., [1, 3, 5])
    
    Returns:
        Dictionary with top-N accuracy metrics
    """
    n_samples = len(y_true)
    
    # Convert y_true to numpy array
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    
    results = {}
    
    for top_n in top_n_values:
        correct_predictions = 0
        total_samples = n_samples
        
        for i in range(total_samples):
            true_label = y_true_np[i]
            # Get top N predicted classes
            top_n_indices = np.argsort(y_pred_proba[i])[-top_n:][::-1]
            
            # Check if true label is in top N predictions
            if true_label in top_n_indices:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        results[f'top_{top_n}_accuracy'] = accuracy
    
    return results

def evaluate_confidence_distribution(y_true, y_pred_proba, label_encoder):
    """
    Analyze confidence levels for correct vs incorrect predictions.
    This helps understand how well confidence scores reflect actual accuracy.
    """
    n_samples = len(y_true)
    
    # Convert y_true to numpy array
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    
    confidences = []
    is_correct = []
    
    for i in range(n_samples):
        true_label = y_true_np[i]
        predicted_label = np.argmax(y_pred_proba[i])
        predicted_confidence = y_pred_proba[i][predicted_label]
        
        confidences.append(predicted_confidence)
        is_correct.append(true_label == predicted_label)
    
    # Calculate confidence statistics
    if confidences:
        correct_confidences = [c for c, correct in zip(confidences, is_correct) if correct]
        incorrect_confidences = [c for c, correct in zip(confidences, is_correct) if not correct]
        
        return {
            'mean_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0,
            'mean_confidence_incorrect': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'confidence_gap': np.mean(correct_confidences) - np.mean(incorrect_confidences) if correct_confidences and incorrect_confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0
        }
    else:
        return {
            'mean_confidence_correct': 0,
            'mean_confidence_incorrect': 0,
            'confidence_gap': 0,
            'confidence_std': 0
        }



# ==============================================================
# TRAIN ALL MODELS WITH ENHANCED METRICS
# ==============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
    'Random Forest': best_rf, 
}

metrics = {}
confusion_matrices = {}
cv_scores = {}  # Store cross-validation scores

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS WITH RANKING METRICS")
print("="*80)

for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    
    # Train model
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        X_for_pred = X_test_scaled
        X_train_for_cv = X_train_scaled
    else:
        model.fit(X_train, y_train)
        X_for_pred = X_test
        X_train_for_cv = X_train
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_for_pred)
        y_pred = model.predict(X_for_pred)
    else:
        # For models without probability (rare)
        y_pred = model.predict(X_for_pred)
        y_pred_proba = None
    
    # ==============================================================
    # ENHANCED METRICS CALCULATION
    # ==============================================================
    
    # Basic metrics
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    # Cross-validation score (5-fold)
    try:
        cv_score = cross_val_score(model, X_train_for_cv, y_train, cv=5, scoring='accuracy')
        cv_scores[name] = cv_score
    except Exception as e:
        print(f"  ⚠ Cross-validation failed for {name}: {e}")
        cv_scores[name] = None
    
    # Ranking metrics (CRITICAL for endpoint usage)
    ranking_metrics = {}
    if y_pred_proba is not None:
        ranking_metrics = evaluate_ranking_metrics(
            y_test, y_pred_proba, le, k_values=[1, 3, 5, 10]
        )
        
        # Top-N accuracy
        top_n_metrics = evaluate_top_n_performance(
            y_test, y_pred_proba, le, top_n_values=[1, 3, 5, 10]
        )
        
        # Confidence analysis
        confidence_metrics = evaluate_confidence_distribution(y_test, y_pred_proba, le)
    else:
        ranking_metrics = {}
        top_n_metrics = {}
        confidence_metrics = {}
    
    # Calculate per-class metrics for the report
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(le.classes_)), zero_division=0
    )
    
    # Store comprehensive metrics
    metrics[name] = {
        # Basic metrics
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        
        # Cross-validation
        'cv_mean_accuracy': cv_score.mean() if cv_scores[name] is not None else None,
        'cv_std_accuracy': cv_score.std() if cv_scores[name] is not None else None,
        
        # Ranking metrics (MOST IMPORTANT for endpoint)
        **ranking_metrics,
        **top_n_metrics,
        
        # Confidence metrics
        **confidence_metrics,
        
        # Per-class metrics summary
        'min_class_f1': float(np.min(f1_per_class[f1_per_class > 0])) if np.any(f1_per_class > 0) else 0,
        'max_class_f1': float(np.max(f1_per_class)),
        'avg_class_f1': float(np.mean(f1_per_class[f1_per_class > 0])) if np.any(f1_per_class > 0) else 0,
        
        # Detailed reports
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=le.classes_, 
            output_dict=True,
            zero_division=0
        ),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        }
    }
    
    print(f"  ✓ Accuracy: {metrics[name]['accuracy']:.4f}") 
    print(f"  ✓ MRR: {metrics[name].get('mrr', 0):.4f}")
    print(f"  ✓ Precision@3: {metrics[name].get('precision@3', 0):.4f}")
    print(f"  ✓ Hit Rate@3: {metrics[name].get('hit_rate@3', 0):.4f}")  
    
    if metrics[name]['cv_mean_accuracy'] is not None:
        print(f"  ✓ CV Mean Accuracy: {metrics[name]['cv_mean_accuracy']:.4f}")
    else:
        print(f"  ✓ CV Mean Accuracy: N/A")

    # ==============================================================
    # SAVE ENHANCED MODEL ARTIFACTS
    # ==============================================================
    model_filename = name.lower().replace(' ', '_')
    
    model_artifacts = {
        'model': model,
        'model_name': name,
        'model_type': type(model).__name__,
        'feature_columns': X.columns.tolist(),
        'crop_classes': le.classes_.tolist(),
        'label_encoder': le,
        'performance_metrics': metrics[name],
        'training_params': model.get_params() if hasattr(model, 'get_params') else {},
        'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING if name in ['Random Forest' ] else None,
        'random_seed': RANDOM_SEED,
        'feature_importance': getattr(model, 'feature_importances_', None),
        'is_probabilistic': hasattr(model, 'predict_proba'),
        'supports_ranking': y_pred_proba is not None,
        'ranking_metrics_available': y_pred_proba is not None
    }
    
    # Add scaler for logistic regression
    if name == 'Logistic Regression':
        model_artifacts['scaler'] = scaler
        model_artifacts['requires_scaling'] = True
    else:
        model_artifacts['requires_scaling'] = False
    
    # Save the model
    model_path = f'models/individual/{model_filename}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    # Save enhanced metrics as JSON
    metrics_path = f'models/individual/{model_filename}_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types
        json_metrics = {}
        for key, value in metrics[name].items():
            if isinstance(value, (np.float32, np.float64)):
                json_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                json_metrics[key] = value
            else:
                json_metrics[key] = value
        
        json.dump(json_metrics, f, indent=2)
    
    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Enhanced metrics saved to: {metrics_path}")

# ============================================================================
# TOP-N RECOMMENDATION MODEL COMPARISON (FOR CROP RECOMMENDATION)
# ============================================================================
print("\n" + "="*80)
print("TOP-N RECOMMENDATION MODEL COMPARISON")
print("="*80)
print("Focusing on metrics relevant for multi-crop recommendation systems")
print("-"*80)

# Define the comparison metrics for recommendation systems
comparison_data = []

for model_name, model in models.items():
    # Get predictions and probabilities
    if model_name == 'Logistic Regression':
        X_for_pred = X_test_scaled
        X_train_for_cv = X_train_scaled
    else:
        X_for_pred = X_test
        X_train_for_cv = X_train
    
    # Get probability predictions for ranking
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_for_pred)
    else:
        y_pred_proba = None
        print(f"⚠ Warning: {model_name} doesn't support probability predictions")
        continue
    
    # Calculate ranking metrics
    ranking_metrics = evaluate_ranking_metrics(
        y_test, y_pred_proba, le, k_values=[1, 3, 5]
    )
    
    top_n_metrics = evaluate_top_n_performance(
        y_test, y_pred_proba, le, top_n_values=[3, 5, 10]
    )
    
    # Cross-validation for stability check
    cv_score = cross_val_score(model, X_train_for_cv, y_train, cv=5, scoring='accuracy')
    
    # Store comparison data
    model_info = {
        'Model': model_name,
        # Primary metrics for recommendation systems
        'MRR': ranking_metrics.get('mrr', 0),           # Mean Reciprocal Rank
        'MAP': ranking_metrics.get('map', 0),           # Mean Average Precision
        'Top-3 Acc': top_n_metrics.get('top_3_accuracy', 0),
        'Top-5 Acc': top_n_metrics.get('top_5_accuracy', 0),
        'Precision@3': ranking_metrics.get('precision@3', 0),
        'Recall@3': ranking_metrics.get('recall@3', 0),
        'Hit Rate@3': ranking_metrics.get('hit_rate@3', 0),
        
        # Secondary metrics
        'Accuracy': accuracy_score(y_test, model.predict(X_for_pred)),
        'F1-Score': f1_score(y_test, model.predict(X_for_pred), average='weighted'),
        
        # Stability metrics
        'CV Mean': cv_score.mean(),
        'CV Std': cv_score.std(),
        
        # Model characteristics
        'Supports Ranking': True,
        'Type': type(model).__name__
    }
    
    comparison_data.append(model_info)

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Sort by Top-3 Accuracy (most relevant for recommending 3 crops)
comparison_df = comparison_df.sort_values('Top-3 Acc', ascending=False)

# Display the comparison table
print("\n📊 MODEL COMPARISON FOR TOP-N CROP RECOMMENDATION:")
print("="*120)

# Format and display the table
formatted_df = comparison_df.copy()
formatted_df = formatted_df[[
    'Model', 'Top-3 Acc', 'Top-5 Acc', 'MRR', 'MAP', 
    'Precision@3', 'Recall@3', 'Hit Rate@3', 'CV Mean'
]]

# Format percentages
for col in formatted_df.columns:
    if col != 'Model':
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")

print(formatted_df.to_string(index=False))
print("\n" + "-"*120)

# Find best models for different use cases
print("\n🏆 BEST MODELS FOR DIFFERENT RECOMMENDATION SCENARIOS:")
print("-"*60)

# Best for Top-3 recommendations (most common use case)
best_top3 = comparison_df.iloc[0]
print(f"1. Best for Top-3 Recommendations:")
print(f"   Model: {best_top3['Model']}")
print(f"   Top-3 Accuracy: {best_top3['Top-3 Acc']:.3f}")
print(f"   MRR: {best_top3['MRR']:.3f} (higher = correct crop appears higher in list)")

# Best for ranking quality (MRR)
best_mrr = comparison_df.loc[comparison_df['MRR'].idxmax()]
print(f"\n2. Best Ranking Quality (MRR):")
print(f"   Model: {best_mrr['Model']}")
print(f"   MRR: {best_mrr['MRR']:.3f}")
print(f"   This model places the correct crop highest in the recommendation list")

# Best for Top-5 recommendations
best_top5 = comparison_df.loc[comparison_df['Top-5 Acc'].idxmax()]
print(f"\n3. Best for Top-5 Recommendations:")
print(f"   Model: {best_top5['Model']}")
print(f"   Top-5 Accuracy: {best_top5['Top-5 Acc']:.3f}")

# Most stable model (lowest CV std)
most_stable = comparison_df.loc[comparison_df['CV Std'].idxmin()]
print(f"\n4. Most Stable Model (lowest variance):")
print(f"   Model: {most_stable['Model']}")
print(f"   CV Std: {most_stable['CV Std']:.4f}")
print(f"   CV Mean Accuracy: {most_stable['CV Mean']:.3f}")

# Practical recommendation
print(f"\n💡 PRACTICAL RECOMMENDATION:")
print("-"*40)
print(f"For a crop recommendation endpoint that provides 3-5 crop suggestions:")
print(f"✓ Use: {best_top3['Model']}")
print(f"✓ Because: Highest Top-3 accuracy ({best_top3['Top-3 Acc']:.3f})")
print(f"✓ MRR: {best_top3['MRR']:.3f} means users see correct crop quickly")
print(f"✓ Provides good balance of accuracy and ranking quality")

# ============================================================================
# VISUALIZATION: Model Comparison for Top-N Recommendation
# ============================================================================
if CREATE_VISUALIZATIONS:
    print("\n" + "="*80)
    print("GENERATING MODEL COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Create visualization directory
    os.makedirs('visualizations/model_comparison', exist_ok=True)
    
    # 1. Radar chart for multi-dimensional comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Select key metrics for radar chart
    radar_metrics = ['Top-3 Acc', 'MRR', 'Precision@3', 'Recall@3', 'CV Mean', 'Top-5 Acc']
    n_metrics = len(radar_metrics)
    
    # Prepare angles for each metric
    angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]  # Close the loop
    
    # Plot each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in radar_metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=row['Model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Radar Chart\n(Top-N Recommendation Metrics)', 
                 fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison/radar_chart.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/model_comparison/radar_chart.png")
    plt.close()
    
    # 2. Bar chart comparison for Top-N accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-N accuracy comparison
    models_list = comparison_df['Model'].tolist()
    top3_acc = comparison_df['Top-3 Acc'].tolist()
    top5_acc = comparison_df['Top-5 Acc'].tolist()
    
    x = np.arange(len(models_list))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, top3_acc, width, label='Top-3 Accuracy', 
                    color='lightblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, top5_acc, width, label='Top-5 Accuracy', 
                    color='lightgreen', edgecolor='black')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Top-N Recommendation Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MRR vs Precision@3 comparison
    mrr_values = comparison_df['MRR'].tolist()
    precision3_values = comparison_df['Precision@3'].tolist()
    
    bars3 = ax2.bar(x - width/2, mrr_values, width, label='MRR', 
                    color='salmon', edgecolor='black')
    bars4 = ax2.bar(x + width/2, precision3_values, width, label='Precision@3', 
                    color='gold', edgecolor='black')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Score')
    ax2.set_title('Ranking Quality Metrics (MRR vs Precision@3)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_list, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison/topn_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/model_comparison/topn_comparison.png")
    plt.close()
    
    # 3. Heatmap of all comparison metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = comparison_df.set_index('Model')
    heatmap_metrics = ['Top-3 Acc', 'Top-5 Acc', 'MRR', 'MAP', 
                      'Precision@3', 'Recall@3', 'Hit Rate@3', 'CV Mean']
    
    # Select and sort the data
    heatmap_values = heatmap_data[heatmap_metrics].values
    sns.heatmap(heatmap_values, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Score'},
                xticklabels=heatmap_metrics,
                yticklabels=heatmap_data.index,
                ax=ax)
    
    ax.set_title('Comprehensive Model Comparison Heatmap', fontsize=14, pad=20)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Models')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison/comparison_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/model_comparison/comparison_heatmap.png")
    plt.close()
    
    print(f"\n✅ Model comparison visualizations saved to: visualizations/model_comparison/")

# Save the comparison DataFrame for later use
comparison_df.to_csv('models/model_comparison_results.csv', index=False)
comparison_df.to_json('models/model_comparison_results.json', indent=2, orient='records')

print(f"\n📁 Model comparison results saved:")
print(f"   • CSV: models/model_comparison_results.csv")
print(f"   • JSON: models/model_comparison_results.json")

# ============================================================================
# FINAL RECOMMENDATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL MODEL SELECTION FOR CROP RECOMMENDATION ENDPOINT")
print("="*80)

# Identify the best model for practical deployment
best_overall = comparison_df.iloc[0]  # Already sorted by Top-3 Accuracy

print(f"\n🎯 RECOMMENDED MODEL FOR DEPLOYMENT:")
print(f"   Model: {best_overall['Model']}")
print(f"   Type: {best_overall['Type']}")
print(f"\n📊 KEY PERFORMANCE METRICS:")
print(f"   • Top-3 Accuracy: {best_overall['Top-3 Acc']:.3f}")
print(f"   • MRR: {best_overall['MRR']:.3f} (ranking quality)")
print(f"   • Top-5 Accuracy: {best_overall['Top-5 Acc']:.3f}")
print(f"   • Cross-validation consistency: {best_overall['CV Mean']:.3f} ± {best_overall['CV Std']:.4f}")

print(f"\n💡 WHAT THIS MEANS FOR YOUR ENDPOINT:")
print(f"   1. When users ask for recommendations, this model will:")
print(f"      • Include the correct crop in top 3 suggestions {best_overall['Top-3 Acc']*100:.1f}% of time")
print(f"      • Place the correct crop high in the list (MRR)")
print(f"   2. Stable performance across different data (low CV std)")
print(f"   3. Suitable for recommending 3-5 crops to users")

print(f"\n🚀 DEPLOYMENT READY:")
print(f"   Use the model at: models/individual/{best_overall['Model'].lower().replace(' ', '_')}.pkl")
print(f"   See full metrics at: models/individual/{best_overall['Model'].lower().replace(' ', '_')}_metrics.json")

# ============================================================================
# ENHANCED VISUALIZATIONS FOR RANKING METRICS
# ============================================================================
if CREATE_VISUALIZATIONS:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR RANKING METRICS")
    print("="*80)
    
    # ==============================================================
    # VISUALIZATION 4: Ranking Metrics Comparison
    # ==============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    ranking_metric_groups = [
        ('MRR & MAP', ['mrr', 'map']),
        ('Precision@K', ['precision@1', 'precision@3', 'precision@5']),
        ('Recall@K', ['recall@1', 'recall@3', 'recall@5']),
        ('Hit Rate@K', ['hit_rate@1', 'hit_rate@3', 'hit_rate@5']),
        ('Top-N Accuracy', ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']),
        ('Confidence Gap', ['confidence_gap'])
    ]
    
    models_list = list(metrics.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
    
    for idx, (title, metric_keys) in enumerate(ranking_metric_groups):
        if idx >= len(axes):
            break
            
        available_metrics = []
        metric_values = []
        
        for model_name in models_list:
            model_metrics = metrics[model_name]
            values = []
            for key in metric_keys:
                if key in model_metrics:
                    values.append(model_metrics[key])
                else:
                    values.append(0)
            if any(v != 0 for v in values):
                available_metrics.append(model_name)
                metric_values.append(values)
        
        if available_metrics:
            x = np.arange(len(metric_keys))
            width = 0.8 / len(available_metrics)
            
            for i, model_name in enumerate(available_metrics):
                offset = (i - len(available_metrics)/2) * width + width/2
                axes[idx].bar(x + offset, metric_values[i], width, 
                             label=model_name, color=colors[i], alpha=0.8)
            
            axes[idx].set_xlabel('Metric')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{title}')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([k.replace('@', '@').replace('_', ' ') 
                                      for k in metric_keys], rotation=45, ha='right')
            axes[idx].legend(loc='upper right', fontsize='small')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_ylim([0, 1])
    
    plt.suptitle('Ranking Metrics Comparison for Recommendation Systems', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/04_ranking_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/04_ranking_metrics_comparison.png")
    plt.close()
    
  
    # ==============================================================
    # VISUALIZATION 6: Feature Importance for Ranking Performance
    # ==============================================================
    # Find best model by MRR (most relevant for ranking)
    best_mrr_model = max(models_list, key=lambda x: metrics[x].get('mrr', 0))
    
    if hasattr(models[best_mrr_model], 'feature_importances_'):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Feature Importance
        importances = models[best_mrr_model].feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_cols[i] for i in indices]
        sorted_importances = importances[indices]
        
        axes[0].barh(range(len(sorted_importances)), sorted_importances, 
                    color='teal', alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(sorted_importances)))
        axes[0].set_yticklabels(sorted_features)
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title(f'Feature Importance: {best_mrr_model}\n(MRR: {metrics[best_mrr_model].get("mrr", 0):.3f})')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        total = np.sum(sorted_importances)
        for i, v in enumerate(sorted_importances):
            axes[0].text(v + 0.01, i, f'{v/total*100:.1f}%', 
                        va='center', fontsize=9)
        
        # Subplot 2: Feature Importance vs Ranking Performance
        # Create a hypothetical analysis (in real scenario, you'd do permutation importance)
        feature_performance = {}
        
        # Simple correlation between feature values and MRR proxy
        # (This is illustrative - in practice you'd do proper analysis)
        for feature in feature_cols:
            # Calculate feature variance (proxy for importance)
            feature_variance = np.var(X_test[feature])
            feature_performance[feature] = feature_variance
        
        # Sort by performance
        sorted_perf = sorted(feature_performance.items(), key=lambda x: x[1], reverse=True)
        features_sorted, perf_sorted = zip(*sorted_perf)
        
        axes[1].barh(range(len(perf_sorted)), perf_sorted, 
                    color='orange', alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(perf_sorted)))
        axes[1].set_yticklabels(features_sorted)
        axes[1].set_xlabel('Feature Variance (Proxy for Discriminative Power)')
        axes[1].set_title('Feature Discriminative Power\n(Higher variance often helps ranking)')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'Feature Analysis for Best Ranking Model: {best_mrr_model}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('visualizations/06_feature_importance_ranking.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: visualizations/06_feature_importance_ranking.png (Best by MRR: {best_mrr_model})")
        plt.close()
    
    # ==============================================================
    # VISUALIZATION 7: Precision-Recall Tradeoff at Different K
    # ==============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models_to_plot = models_list[:4]  # Plot first 4 models
    
    for idx, model_name in enumerate(models_to_plot):
        if idx >= len(axes):
            break
            
        model_metrics = metrics[model_name]
        
        # Extract precision and recall at different K
        k_values = [1, 3, 5, 10]
        precision_values = []
        recall_values = []
        
        for k in k_values:
            precision_key = f'precision@{k}'
            recall_key = f'recall@{k}'
            
            if precision_key in model_metrics and recall_key in model_metrics:
                precision_values.append(model_metrics[precision_key])
                recall_values.append(model_metrics[recall_key])
        
        if precision_values and recall_values:
            # Plot precision-recall curve
            axes[idx].plot(k_values[:len(precision_values)], precision_values, 
                          'o-', label='Precision', linewidth=2, markersize=8)
            axes[idx].plot(k_values[:len(recall_values)], recall_values, 
                          's-', label='Recall', linewidth=2, markersize=8)
            
            axes[idx].set_xlabel('K (Number of Recommendations)')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{model_name}\nPrecision & Recall at K')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(k_values[:len(precision_values)])
            axes[idx].set_ylim([0, 1])
            
            # Add MRR value
            if 'mrr' in model_metrics:
                axes[idx].text(0.05, 0.95, f'MRR: {model_metrics["mrr"]:.3f}',
                              transform=axes[idx].transAxes,
                              fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Precision-Recall Tradeoff at Different Recommendation Depths', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/07_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/07_precision_recall_tradeoff.png")
    plt.close()




    
    # ============================================================================
# VISUALIZATION 8: Confusion Matrices for All Models
# ============================================================================
if CREATE_VISUALIZATIONS:
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
    print("="*80)
    
    # Create a directory for confusion matrices
    os.makedirs('visualizations/confusion_matrices', exist_ok=True)
    
    # Create confusion matrix for each model
    for model_name, cm in confusion_matrices.items():
        plt.figure(figsize=(14, 12))
        
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create a mask for values > 0 (to hide zeros in visualization)
        mask = cm == 0
        
        # Plot with annotations
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   cbar_kws={'label': 'Proportion'},
                   mask=mask,
                   vmin=0, vmax=1)
        
        plt.title(f'Confusion Matrix: {model_name}\n(Normalized by True Labels)', 
                 fontsize=16, pad=20)
        plt.xlabel('Predicted Crop', fontsize=12)
        plt.ylabel('True Crop', fontsize=12)
        
        # Set tick labels with crop names
        tick_labels = le.classes_
        plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, 
                  rotation=45, ha='right', fontsize=8)
        plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, 
                  rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        # Save the confusion matrix
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        save_path = f'visualizations/confusion_matrices/{filename}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    # ==============================================================
    # VISUALIZATION 9: Side-by-Side Confusion Matrix Comparison
    # ==============================================================
    print("\nGenerating comparison visualization...")
    
    # Create a grid of confusion matrices for comparison
    n_models = len(confusion_matrices)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_models > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Proportion', rotation=270, labelpad=15)
        
        # Set title
        ax.set_title(f'{model_name}\nAccuracy: {metrics[model_name]["accuracy"]:.3f}', 
                    fontsize=12, pad=10)
        
        # Set tick labels (only show every 5th label to avoid clutter)
        tick_interval = max(1, len(le.classes_) // 10)
        tick_indices = np.arange(0, len(le.classes_), tick_interval)
        tick_labels = le.classes_[tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Add grid
        ax.set_xticks(np.arange(-.5, len(le.classes_), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(le.classes_), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2, alpha=0.3)
        
        # Remove minor tick labels
        ax.tick_params(which="minor", bottom=False, left=False)
    
    # Hide unused subplots
    for idx in range(len(confusion_matrices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrix Comparison Across Models', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices/confusion_matrices_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/confusion_matrices/confusion_matrices_comparison.png")
    plt.close()
    
    # ==============================================================
    # VISUALIZATION 10: Per-Class Performance Analysis
    # ==============================================================
    print("\nGenerating per-class performance analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    # Define metrics to visualize
    class_metrics = ['precision', 'recall', 'f1', 'support']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Support']
    
    for idx, (metric_key, metric_name) in enumerate(zip(class_metrics, metric_names)):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        crops = le.classes_
        n_crops = len(crops)
        models_list = list(metrics.keys())
        n_models = len(models_list)
        
        # Set bar positions
        bar_width = 0.8 / n_models
        x_positions = np.arange(n_crops)
        
        # Colors for different models
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        # Plot bars for each model
        for model_idx, model_name in enumerate(models_list):
            # Get per-class metrics
            model_metric_data = metrics[model_name]['per_class_metrics'][metric_key]
            
            # Calculate offset for each model's bars
            offset = (model_idx - n_models/2) * bar_width + bar_width/2
            
            # Plot bars
            bars = ax.bar(x_positions + offset, model_metric_data, 
                         width=bar_width, color=colors[model_idx], 
                         alpha=0.7, label=model_name, edgecolor='black', linewidth=0.5)
            
            # Add value labels on top of bars (only for notable values)
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.5:  # Only label high values to avoid clutter
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=7, rotation=90)
        
        # Customize subplot
        ax.set_xlabel('Crop', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f'Per-Crop {metric_name} by Model', fontsize=12, pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(crops, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend (only for first subplot to avoid repetition)
        if idx == 0:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
    
    plt.suptitle('Detailed Per-Crop Performance Analysis', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices/per_class_performance_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/confusion_matrices/per_class_performance_analysis.png")
    plt.close()
 
 
# ============================================================================
# SAVE ENHANCED SUMMARY WITH RANKING METRICS
# ============================================================================
# Save all models together in one file
all_models_artifacts = {
    'models': models,
    'metrics': metrics,
    'label_encoder': le,
    'scaler': scaler,
    'feature_columns': X.columns.tolist(),
    'crop_classes': le.classes_.tolist(),
    'best_model_by_mrr': max(models_list, key=lambda x: metrics[x].get('mrr', 0)),
    'best_model_by_accuracy': max(models_list, key=lambda x: metrics[x]['accuracy']),
    'best_model_by_top3': max(models_list, key=lambda x: metrics[x].get('top_3_accuracy', 0)),
    'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING,
    'random_seed': RANDOM_SEED,
    'training_timestamp': str(pd.Timestamp.now())
}

os.makedirs('precomputation', exist_ok=True)
with open('precomputation/all_models_artifacts.pkl', 'wb') as f:
    pickle.dump(all_models_artifacts, f)

# Save comprehensive summary
summary = {
    'dataset_info': {
        'total_samples': len(df),
        'num_crops': len(le.classes_),
        'crops': le.classes_.tolist(),
        'features': X.columns.tolist(),
        'train_test_split': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_percentage': 0.3
        }
    },
    'model_performance': {},
    'model_selection_recommendation': {},
    'ranking_metrics_summary': {},
    'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING,
    'random_seed': RANDOM_SEED,
    'visualizations_created': CREATE_VISUALIZATIONS,
    'training_date': str(pd.Timestamp.now().date()),
    'endpoint_usage_note': 'Models are used individually with ranking-based recommendations'
}

# Add model performance
for model_name, model_metrics in metrics.items():
    summary['model_performance'][model_name] = {
        # Basic metrics
        'accuracy': float(model_metrics['accuracy']),
        'precision': float(model_metrics['precision']),
        'recall': float(model_metrics['recall']),
        'f1_score': float(model_metrics['f1_score']),
        
        # Ranking metrics
        'mrr': float(model_metrics.get('mrr', 0)),
        'map': float(model_metrics.get('map', 0)),
        'top_3_accuracy': float(model_metrics.get('top_3_accuracy', 0)),
        'precision@3': float(model_metrics.get('precision@3', 0)),
        'hit_rate@3': float(model_metrics.get('hit_rate@3', 0)),
        
        # Cross-validation
        'cv_mean_accuracy': float(model_metrics['cv_mean_accuracy']) if model_metrics['cv_mean_accuracy'] else None,
        
        # Consistency
        'confidence_gap': float(model_metrics.get('confidence_gap', 0)),
        'avg_class_f1': float(model_metrics['avg_class_f1'])
    }

# Ranking metrics summary
summary['ranking_metrics_summary'] = {
    'best_mrr_model': max(models_list, key=lambda x: metrics[x].get('mrr', 0)),
    'best_mrr_value': float(metrics[max(models_list, key=lambda x: metrics[x].get('mrr', 0))].get('mrr', 0)),
    'best_top3_model': max(models_list, key=lambda x: metrics[x].get('top_3_accuracy', 0)),
    'best_top3_value': float(metrics[max(models_list, key=lambda x: metrics[x].get('top_3_accuracy', 0))].get('top_3_accuracy', 0)),
    'average_mrr_across_models': float(np.mean([metrics[m].get('mrr', 0) for m in models_list])),
    'average_top3_accuracy': float(np.mean([metrics[m].get('top_3_accuracy', 0) for m in models_list]))
}

# Model selection recommendations
summary['model_selection_recommendation'] = {
    'best_overall_accuracy': max(models_list, key=lambda x: metrics[x]['accuracy']),
    'best_for_ranking_mrr': max(models_list, key=lambda x: metrics[x].get('mrr', 0)),
    'best_for_top3_recommendations': max(models_list, key=lambda x: metrics[x].get('top_3_accuracy', 0)),
    'best_confidence_calibration': max(models_list, key=lambda x: metrics[x].get('confidence_gap', 0)),
    'recommendation_for_endpoint': max(models_list, key=lambda x: metrics[x].get('mrr', 0)),  # MRR is best for ranking
    'explanation': 'MRR is optimal for recommendation systems where ranking quality matters most'
}

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ============================================================================
# FINAL OUTPUT WITH RANKING METRICS
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - WITH RANKING METRICS")
print("="*80)

# Detailed ranking metrics table
print(f"\n{'Model':<25} {'Accuracy':<8} {'MRR':<8} {'Prec@3':<8} {'Recall@3':<8} {'Hit@3':<8} {'Top-3 Acc':<8}")
print("-"*90)
for model_name, metric in metrics.items():
    print(f"{model_name:<25} "
          f"{metric['accuracy']:<8.4f} "
          f"{metric.get('mrr', 0):<8.4f} "
          f"{metric.get('precision@3', 0):<8.4f} "
          f"{metric.get('recall@3', 0):<8.4f} "
          f"{metric.get('hit_rate@3', 0):<8.4f} "
          f"{metric.get('top_3_accuracy', 0):<8.4f}")

# Summary
print(f"\n📊 RANKING METRICS SUMMARY:")
print(f"   Best MRR: {summary['ranking_metrics_summary']['best_mrr_model']} "
      f"({summary['ranking_metrics_summary']['best_mrr_value']:.4f})")
print(f"   Best Top-3 Accuracy: {summary['ranking_metrics_summary']['best_top3_model']} "
      f"({summary['ranking_metrics_summary']['best_top3_value']:.4f})")
print(f"   Average MRR across models: {summary['ranking_metrics_summary']['average_mrr_across_models']:.4f}")
print(f"   Average Top-3 Accuracy: {summary['ranking_metrics_summary']['average_top3_accuracy']:.4f}")

print(f"\n🏆 RECOMMENDATION FOR ENDPOINT:")
print(f"   {summary['model_selection_recommendation']['recommendation_for_endpoint']} "
      f"(MRR: {metrics[summary['model_selection_recommendation']['recommendation_for_endpoint']].get('mrr', 0):.4f})")
print(f"   MRR measures how high the correct crop appears in recommendations")
print(f"   Higher MRR = better ranking quality for users")

print(f"\n✅ All models and metrics saved:")
print(f"   • Individual models: models/individual/")
print(f"   • Complete summary: models/model_summary.json")
print(f"   • All artifacts: precomputation/all_models_artifacts.pkl")

if CREATE_VISUALIZATIONS:
    print(f"   • Visualizations: visualizations/ (7 ranking-focused charts)")
    print(f"   • Confusion Matrices: visualizations/confusion_matrices/ (4 detailed analysis charts)")
    print(f"   • Model Comparison: visualizations/model_comparison/ (3 comparison charts)")

print("\n" + "="*80)
print("READY FOR ENDPOINT DEPLOYMENT")
print("="*80)