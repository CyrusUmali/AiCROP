import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ------------------ Setup and Initialization ------------------

# Paths (relative to script location)
MODELS_DIR = Path("models/individual")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")
ANALYSIS_OUTPUT_DIR = Path("dataset_analysis")

# Global variables
models: Dict[str, dict] = {}
label_encoder = None
feature_columns = None
crop_classes = None
df = None

# ------------------ MANUAL TEST PARAMETERS ------------------
# MODIFY THESE VALUES TO TEST DIFFERENT SCENARIOS

MANUAL_TEST_PARAMS = {
    'soil_ph': 6.9,
    'fertility_ec': 540.0,
    'humidity': 76.0,
    'sunlight': 2900.0,
    'soil_temp': 29.1,
    'soil_moisture': 93.0
}

# Crop to test
CROP_TO_TEST = "String Bean"  # Change this to any crop name

# Confidence threshold
THRESHOLD = 0.7

# MODELS TO USE - ADD THIS CONFIGURATION
# Set to None to use all models, or specify a list of model names
MODELS_TO_USE = ["XGBoost"]  # Example: ["RandomForestClassifier", "KNeighborsClassifier", "SVC"]

# ------------------ NEW DATASET ANALYSIS FUNCTIONS ------------------

def feature_score(value, min_val, max_val):
    """Calculate feature score"""
    if min_val <= value <= max_val:
        return 1.0
    range_width = max_val - min_val if max_val != min_val else 1
    if value < min_val:
        return max(0.0, 1 - abs(value - min_val) / range_width)
    else:
        return max(0.0, 1 - abs(value - max_val) / range_width)


def analyze_dataset_for_crop(crop_name: str, test_params: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze how test parameters compare to overall dataset distribution
    """
    if df is None:
        return {}
    
    crop_data = df[df['label'] == crop_name]
    other_crops_data = df[df['label'] != crop_name]
    
    analysis = {
        'crop': crop_name,
        'crop_sample_count': len(crop_data),
        'total_samples': len(df),
        'crop_percentage': len(crop_data) / len(df) * 100,
        'parameter_analysis': {},
        'outlier_detection': {},
        'feature_correlations': {},
        'parameter_combinations': {},
        'visualization_paths': []
    }
    
    # Ensure output directory exists
    ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Analyze each parameter individually
    for param in feature_columns:
        if param not in test_params:
            continue
            
        test_value = test_params[param]
        crop_param_data = crop_data[param]
        other_param_data = other_crops_data[param]
        all_param_data = df[param]
        
        # Calculate percentiles
        crop_percentile = (crop_param_data <= test_value).mean() * 100
        overall_percentile = (all_param_data <= test_value).mean() * 100
        other_percentile = (other_param_data <= test_value).mean() * 100
        
        # Calculate z-scores
        crop_z = (test_value - crop_param_data.mean()) / crop_param_data.std() if crop_param_data.std() > 0 else 0
        overall_z = (test_value - all_param_data.mean()) / all_param_data.std() if all_param_data.std() > 0 else 0
        
        # Check if value is outlier (beyond 2 standard deviations)
        is_crop_outlier = abs(crop_z) > 2
        is_overall_outlier = abs(overall_z) > 2
        
        # Calculate density at test value (simplified)
        crop_density = len(crop_param_data[(crop_param_data >= test_value * 0.95) & 
                                          (crop_param_data <= test_value * 1.05)]) / len(crop_param_data)
        overall_density = len(all_param_data[(all_param_data >= test_value * 0.95) & 
                                            (all_param_data <= test_value * 1.05)]) / len(all_param_data)
        
        analysis['parameter_analysis'][param] = {
            'test_value': test_value,
            'crop_mean': float(crop_param_data.mean()),
            'crop_std': float(crop_param_data.std()),
            'crop_min': float(crop_param_data.min()),
            'crop_max': float(crop_param_data.max()),
            'overall_mean': float(all_param_data.mean()),
            'overall_std': float(all_param_data.std()),
            'overall_min': float(all_param_data.min()),
            'overall_max': float(all_param_data.max()),
            'crop_percentile': float(crop_percentile),
            'overall_percentile': float(overall_percentile),
            'other_percentile': float(other_percentile),
            'crop_z_score': float(crop_z),
            'overall_z_score': float(overall_z),
            'is_crop_outlier': bool(is_crop_outlier),
            'is_overall_outlier': bool(is_overall_outlier),
            'crop_density_at_value': float(crop_density),
            'overall_density_at_value': float(overall_density),
            'distance_from_crop_mean': float(abs(test_value - crop_param_data.mean())),
            'distance_from_overall_mean': float(abs(test_value - all_param_data.mean()))
        }
        
        # Outlier detection
        if is_overall_outlier:
            analysis['outlier_detection'][param] = {
                'severity': 'high' if abs(overall_z) > 3 else 'medium',
                'z_score': float(overall_z),
                'percentile': float(overall_percentile),
                'description': f"Value is {overall_z:.1f} std devs from overall mean"
            }
    
    # 2. Analyze parameter combinations (multivariate analysis)
    if len(feature_columns) >= 2:
        # Calculate Mahalanobis distance for the crop's data
        try:
            from scipy.spatial.distance import mahalanobis
            from scipy.linalg import inv
            
            # Prepare data for the specific crop
            crop_features = crop_data[feature_columns].values
            if len(crop_features) > len(feature_columns):  # Ensure enough samples
                cov_matrix = np.cov(crop_features, rowvar=False)
                if np.linalg.matrix_rank(cov_matrix) == len(feature_columns):
                    inv_cov_matrix = inv(cov_matrix)
                    mean_vector = np.mean(crop_features, axis=0)
                    
                    # Calculate Mahalanobis distance for test point
                    test_vector = np.array([test_params.get(p, 0) for p in feature_columns])
                    mahal_dist = mahalanobis(test_vector, mean_vector, inv_cov_matrix)
                    
                    # Calculate distances for all crop samples to get distribution
                    mahal_distances = []
                    for point in crop_features:
                        try:
                            dist = mahalanobis(point, mean_vector, inv_cov_matrix)
                            mahal_distances.append(dist)
                        except:
                            continue
                    
                    if mahal_distances:
                        mahal_percentile = (np.array(mahal_distances) <= mahal_dist).mean() * 100
                        analysis['parameter_combinations']['mahalanobis_distance'] = {
                            'distance': float(mahal_dist),
                            'percentile_in_crop': float(mahal_percentile),
                            'is_outlier': mahal_percentile > 95 or mahal_percentile < 5,
                            'description': f"Multivariate distance from crop center: {mahal_dist:.2f}"
                        }
        except Exception as e:
            analysis['parameter_combinations']['mahalanobis_error'] = str(e)
    
    # 3. Calculate feature correlations for the crop
    crop_corr_matrix = crop_data[feature_columns].corr()
    analysis['feature_correlations'] = {
        'matrix': crop_corr_matrix.to_dict(),
        'strong_correlations': []
    }
    
    # Find strong correlations (> 0.7 or < -0.7)
    for i in range(len(feature_columns)):
        for j in range(i+1, len(feature_columns)):
            corr_value = crop_corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                analysis['feature_correlations']['strong_correlations'].append({
                    'feature1': feature_columns[i],
                    'feature2': feature_columns[j],
                    'correlation': float(corr_value)
                })
    
    # 4. Check if parameter combination is rare
    rare_combination_score = 0
    rare_params = []
    for param in feature_columns:
        if param in test_params:
            param_analysis = analysis['parameter_analysis'][param]
            if param_analysis['overall_density_at_value'] < 0.01:  # Less than 1% of data
                rare_combination_score += 1
                rare_params.append(param)
    
    analysis['combination_rarity'] = {
        'score': rare_combination_score,
        'rare_parameters': rare_params,
        'description': f"{rare_combination_score} parameters have values in rare regions (<1% density)"
    }
    
    # 5. Generate visualizations
    try:
        # Create comparison plots for each parameter
        for param in feature_columns[:4]:  # Limit to first 4 parameters for readability
            if param not in test_params:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Distribution comparison
            axes[0].hist(crop_data[param], alpha=0.5, bins=30, label=f'{crop_name} (n={len(crop_data)})', 
                        color='blue', density=True)
            axes[0].hist(other_crops_data[param], alpha=0.5, bins=30, label=f'Other Crops (n={len(other_crops_data)})', 
                        color='red', density=True)
            axes[0].axvline(test_params[param], color='green', linestyle='--', linewidth=2, 
                          label=f'Test Value: {test_params[param]:.1f}')
            axes[0].set_xlabel(param.replace('_', ' ').title())
            axes[0].set_ylabel('Density')
            axes[0].set_title(f'{param.replace("_", " ").title()} Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Box plot comparison
            box_data = [crop_data[param].dropna(), other_crops_data[param].dropna()]
            axes[1].boxplot(box_data, labels=[crop_name, 'Other Crops'])
            axes[1].scatter([1, 2], [crop_data[param].mean(), other_crops_data[param].mean()], 
                          color='red', marker='D', label='Mean')
            axes[1].axhline(test_params[param], color='green', linestyle='--', 
                          label=f'Test: {test_params[param]:.1f}')
            axes[1].set_ylabel(param.replace('_', ' ').title())
            axes[1].set_title(f'{param.replace("_", " ").title()} Box Plot')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = ANALYSIS_OUTPUT_DIR / f"{crop_name}_{param}_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            analysis['visualization_paths'].append(str(plot_path))
            
    except Exception as e:
        analysis['visualization_error'] = str(e)
    
    return analysis


def print_dataset_analysis(analysis: Dict[str, Any]):
    """
    Print detailed dataset analysis results
    """
    if not analysis:
        print("❌ No analysis data available")
        return
    
    print("\n" + "="*80)
    print("📊 DATASET ANALYSIS FOR SPECIFIC CROP")
    print("="*80)
    
    print(f"\n🌱 CROP INFORMATION:")
    print(f"  Crop: {analysis['crop']}")
    print(f"  Samples for this crop: {analysis['crop_sample_count']:,}")
    print(f"  Total dataset samples: {analysis['total_samples']:,}")
    print(f"  Percentage of dataset: {analysis['crop_percentage']:.1f}%")
    
    if analysis['crop_sample_count'] < 100:
        print(f"  ⚠️  WARNING: Small sample size for this crop (<100 samples)")
    
    print(f"\n📈 PARAMETER DISTRIBUTION ANALYSIS:")
    print(f"{'Parameter':<20} {'Test Value':<12} {'Crop Mean':<12} {'Crop Z':<8} {'Overall Z':<10} {'Crop %ile':<10} {'Overall %ile':<12} {'Outlier'}")
    print(f"{'-'*100}")
    
    for param, param_analysis in analysis['parameter_analysis'].items():
        param_name = param.replace('_', ' ').title()
        test_val = param_analysis['test_value']
        crop_mean = param_analysis['crop_mean']
        crop_z = param_analysis['crop_z_score']
        overall_z = param_analysis['overall_z_score']
        crop_percentile = param_analysis['crop_percentile']
        overall_percentile = param_analysis['overall_percentile']
        is_outlier = param_analysis['is_overall_outlier']
        
        outlier_symbol = "⚠️" if is_outlier else "✓"
        
        print(f"{param_name:<20} {test_val:<12.2f} {crop_mean:<12.2f} {crop_z:<8.2f} "
              f"{overall_z:<10.2f} {crop_percentile:<10.1f}% {overall_percentile:<12.1f}% {outlier_symbol}")
    
    # Outlier detection summary
    if analysis['outlier_detection']:
        print(f"\n🚨 OUTLIER DETECTION:")
        for param, outlier_info in analysis['outlier_detection'].items():
            param_name = param.replace('_', ' ').title()
            print(f"  ⚠️  {param_name}: {outlier_info['description']}")
            print(f"     Severity: {outlier_info['severity'].upper()}, "
                  f"Percentile: {outlier_info['percentile']:.1f}%")
    
    # Parameter combinations analysis
    if 'mahalanobis_distance' in analysis.get('parameter_combinations', {}):
        mahal_info = analysis['parameter_combinations']['mahalanobis_distance']
        print(f"\n🔍 MULTIVARIATE ANALYSIS:")
        print(f"  Mahalanobis Distance: {mahal_info['distance']:.2f}")
        print(f"  Percentile in crop distribution: {mahal_info['percentile_in_crop']:.1f}%")
        if mahal_info['is_outlier']:
            print(f"  ⚠️  MULTIVARIATE OUTLIER: This combination of parameters is unusual for this crop")
    
    # Combination rarity
    if analysis['combination_rarity']['score'] > 0:
        print(f"\n🎯 PARAMETER COMBINATION RARITY:")
        print(f"  Score: {analysis['combination_rarity']['score']}/6 parameters in rare regions")
        print(f"  Rare parameters: {', '.join(analysis['combination_rarity']['rare_parameters'])}")
        print(f"  ℹ️  Rare parameter combinations can confuse ML models even if individual parameters seem OK")
    
    # Feature correlations
    if analysis['feature_correlations']['strong_correlations']:
        print(f"\n🔗 STRONG FEATURE CORRELATIONS (|r| > 0.7):")
        for corr in analysis['feature_correlations']['strong_correlations']:
            f1 = corr['feature1'].replace('_', ' ').title()
            f2 = corr['feature2'].replace('_', ' ').title()
            corr_val = corr['correlation']
            direction = "positive" if corr_val > 0 else "negative"
            print(f"  {f1} ↔ {f2}: {corr_val:.3f} ({direction})")
    
    # Density analysis
    print(f"\n📊 DENSITY ANALYSIS (how common are these exact values):")
    for param, param_analysis in analysis['parameter_analysis'].items():
        if param_analysis['overall_density_at_value'] < 0.05:  # Less than 5% density
            param_name = param.replace('_', ' ').title()
            density = param_analysis['overall_density_at_value'] * 100
            print(f"  ⚠️  {param_name}: Only {density:.2f}% of ALL crops have similar values (±5%)")
    
    # Model confidence insights
    print(f"\n💡 INSIGHTS FOR MODEL CONFIDENCE:")
    
    # Check for small sample size
    if analysis['crop_sample_count'] < 50:
        print(f"  • Small training sample ({analysis['crop_sample_count']} samples)")
        print(f"    → Models struggle to learn patterns from limited data")
    
    # Check for many outliers
    outlier_count = sum(1 for p in analysis['parameter_analysis'].values() if p['is_overall_outlier'])
    if outlier_count >= 2:
        print(f"  • {outlier_count} parameters are statistical outliers")
        print(f"    → ML models are less confident with unusual value combinations")
    
    # Check if test values are near extremes
    extreme_params = []
    for param, param_analysis in analysis['parameter_analysis'].items():
        percentile = param_analysis['overall_percentile']
        if percentile < 5 or percentile > 95:
            extreme_params.append(param)
    
    if extreme_params:
        print(f"  • Parameters in extreme percentiles: {', '.join(extreme_params)}")
        print(f"    → Values in the 5th or 95th+ percentile can reduce model confidence")
    
    # Visualization info
    if analysis.get('visualization_paths'):
        print(f"\n📈 VISUALIZATIONS GENERATED:")
        for path in analysis['visualization_paths']:
            print(f"  • {Path(path).name}")
        print(f"  📁 Saved in: {ANALYSIS_OUTPUT_DIR}")
    
    print("\n" + "="*80)


def analyze_model_predictions_for_crop(crop_name: str, test_params: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze what crops the model actually predicts for the given parameters
    """
    if not models:
        return {}
    
    # Create input data
    input_values = [test_params.get(param, 0) for param in feature_columns]
    input_data = pd.DataFrame([input_values], columns=feature_columns)
    
    prediction_analysis = {
        'crop': crop_name,
        'true_crop_idx': label_encoder.transform([crop_name])[0] if label_encoder else -1,
        'model_predictions': {}
    }
    
    for model_name, model_data in models.items():
        model = model_data['model']
        requires_scaling = model_data.get('requires_scaling', False)
        
        # Scale if needed
        if requires_scaling:
            scaler = model_data.get('scaler')
            input_processed = scaler.transform(input_data) if scaler else input_data.values
        else:
            input_processed = input_data.values
        
        # Get all prediction probabilities
        try:
            all_probas = model.predict_proba(input_processed)[0]
        except:
            continue
        
        # Get top 5 predictions
        top_5_indices = np.argsort(all_probas)[-5:][::-1]
        top_5_crops = label_encoder.inverse_transform(top_5_indices)
        top_5_confidences = all_probas[top_5_indices]
        
        # Get predicted class
        predicted_idx = np.argmax(all_probas)
        predicted_crop = label_encoder.inverse_transform([predicted_idx])[0]
        predicted_conf = all_probas[predicted_idx]
        
        prediction_analysis['model_predictions'][model_name] = {
            'top_5_predictions': list(zip(top_5_crops, top_5_confidences)),
            'predicted_crop': predicted_crop,
            'predicted_confidence': float(predicted_conf),
            'true_crop_rank': int(np.where(top_5_indices == prediction_analysis['true_crop_idx'])[0][0] 
                                 if prediction_analysis['true_crop_idx'] in top_5_indices else -1),
            'true_crop_confidence': float(all_probas[prediction_analysis['true_crop_idx']] 
                                         if prediction_analysis['true_crop_idx'] < len(all_probas) else 0)
        }
    
    return prediction_analysis


def print_model_prediction_analysis(analysis: Dict[str, Any]):
    """
    Print model prediction analysis
    """
    if not analysis or not analysis.get('model_predictions'):
        return
    
    print("\n" + "="*80)
    print("🤖 MODEL PREDICTION ANALYSIS")
    print("="*80)
    
    print(f"\n🎯 TRUE CROP: {analysis['crop']}")
    
    for model_name, pred_info in analysis['model_predictions'].items():
        print(f"\n📊 {model_name}:")
        print(f"  Predicted: {pred_info['predicted_crop']} "
              f"(confidence: {pred_info['predicted_confidence']:.2%})")
        
        true_crop_conf = pred_info['true_crop_confidence']
        rank = pred_info['true_crop_rank']
        
        if rank >= 0:
            print(f"  True crop rank: #{rank + 1} (confidence: {true_crop_conf:.2%})")
        else:
            print(f"  ❌ True crop not in top 5 predictions (confidence: {true_crop_conf:.2%})")
        
        print(f"  Top 5 predictions:")
        for i, (crop, conf) in enumerate(pred_info['top_5_predictions'], 1):
            marker = "👑" if i == 1 else "   "
            true_marker = "🎯" if crop == analysis['crop'] else "  "
            print(f"    {i:2d}. {marker} {crop:<25} {true_marker} {conf:.2%}")

def show_available_models() -> List[str]:
    """Show available models and return list of names"""
    all_models = load_all_models()
    print("\n📋 AVAILABLE MODELS:")
    for i, (model_name, model_data) in enumerate(all_models.items(), 1):
        model_type = type(model_data['model']).__name__
        print(f"  {i}. {model_name:<25} ({model_type})")
    return list(all_models.keys())

def load_all_models() -> Dict[str, dict]:
    """Load all models from directory"""
    available_models = {}
    for file_path in MODELS_DIR.glob("*.pkl"):
        if "_metrics" in file_path.name:
            continue
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            model_name = model_data.get('model_name', file_path.stem)
            available_models[model_name] = model_data
    return available_models



def print_results(result: Dict):
    """
    Print test results
    """
    print("\n" + "="*80)
    print("🔬 MANUAL TEST RESULTS")
    print("="*80)
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
        return
    
    # Basic info
    status = "✅ SUITABLE" if result['is_suitable'] else "❌ NOT SUITABLE"
    print(f"\n📋 OVERVIEW")
    print(f"  Crop: {result['crop']}")
    print(f"  Status: {status}")
    print(f"  Final Confidence: {result['final_confidence']:.2%}")
    print(f"  Parameter Confidence: {result['parameter_confidence']:.2%}")
    print(f"  Average Model Confidence: {result['model_confidence_avg']:.2%}")
    print(f"  Threshold: {result['threshold_used']:.0%}")
    
    # Paradox detection
    if result['paradox_detected']:
        print(f"\n⚠️  PARADOX DETECTED!")
        print(f"  All parameters are optimal but model confidence is low ({result['model_confidence_avg']:.2%})")
    
    # Input values
    print(f"\n📊 INPUT VALUES:")
    for param, value in result['input_values'].items():
        print(f"  {param.replace('_', ' ').title():<20}: {value:.2f}")
    
    # Parameter analysis
    print(f"\n🎯 PARAMETER ANALYSIS:")
    print(f"{'Parameter':<20} {'Current':<10} {'Ideal Range':<20} {'Score':<10} {'Status':<10}")
    print(f"{'-'*70}")
    
    param_analysis = result['parameters_analysis']
    for param, analysis in param_analysis.items():
        param_name = param.replace('_', ' ').title()
        current = analysis['current']
        ideal_range = f"{analysis['ideal_min']:.1f}-{analysis['ideal_max']:.1f}"
        score = analysis['feature_score']
        status = analysis['status'].upper()
        
        if status == 'OPTIMAL':
            status_display = f"✅ {status}"
        elif status == 'LOW':
            status_display = f"🔻 {status}"
        else:
            status_display = f"🔺 {status}"
        
        print(f"{param_name:<20} {current:<10.2f} {ideal_range:<20} {score:<10.2f} {status_display:<15}")
    
    # Model confidences
    print(f"\n🤖 MODEL CONFIDENCES:")
    for model_name, confidence in result['ml_confidence'].items():
        model_type = result['model_details'][model_name]['model_type']
        above_thresh = "✅" if confidence >= result['threshold_used'] else "❌"
        print(f"  {model_name:<20} ({model_type:<10}): {confidence:.2%} {above_thresh}")
    
    # Dataset stats
    print(f"\n📈 DATASET STATISTICS FOR '{result['crop']}':")
    print(f"  Dataset samples: {list(result['dataset_stats'].values())[0]['count']}")
    
    # Feature scores
    print(f"\n🎯 FEATURE SCORES:")
    for i, (param, score) in enumerate(zip(feature_columns, result['feature_scores'])):
        param_name = param.replace('_', ' ').title()
        print(f"  {param_name:<20}: {score:.2f} {'✅' if score > 0.8 else '⚠️' if score > 0.5 else '❌'}")
    
    # Feature importance
    print(f"\n⚖️  FEATURE IMPORTANCE (Average across models):")
    avg_importances = []
    for i, param in enumerate(feature_columns):
        importances = []
        for model_name, details in result['model_details'].items():
            if i < len(details['feature_importance']):
                importances.append(details['feature_importance'][i])
        
        if importances:
            avg_imp = np.mean(importances)
            std_imp = np.std(importances)
            avg_importances.append((param, avg_imp, std_imp))
    
    # Sort by importance
    avg_importances.sort(key=lambda x: x[1], reverse=True)
    
    for param, avg_imp, std_imp in avg_importances:
        param_name = param.replace('_', ' ').title()
        bar_length = int(avg_imp * 20 / max(0.001, max([x[1] for x in avg_importances])))
        bar = "█" * bar_length
        print(f"  {param_name:<20} {bar} {avg_imp:.3f} (±{std_imp:.3f})")
    
    # Debug insights
    print(f"\n🔍 DEBUG INSIGHTS:")
    
    # Check if input is near dataset mean
    for param in feature_columns:
        current = result['input_values'][param]
        mean_val = result['dataset_stats'][param]['mean']
        std_val = result['dataset_stats'][param]['std']
        
        if std_val > 0:
            z_score = abs(current - mean_val) / std_val
            if z_score > 2:
                print(f"  ⚠️  {param.replace('_', ' ').title()} is {z_score:.1f} std devs from mean")
        
        # Check if near edges of ideal range
        analysis = param_analysis[param]
        if analysis['in_ideal_range']:
            range_width = analysis['ideal_max'] - analysis['ideal_min']
            if range_width > 0:
                dist_to_min = abs(current - analysis['ideal_min']) / range_width
                dist_to_max = abs(current - analysis['ideal_max']) / range_width
                if dist_to_min < 0.1 or dist_to_max < 0.1:
                    print(f"  ⚠️  {param.replace('_', ' ').title()} near edge of ideal range")
    
    # Model consistency
    model_confs = list(result['ml_confidence'].values())
    if len(model_confs) > 1:
        conf_std = np.std(model_confs)
        if conf_std > 0.3:
            print(f"  ⚠️  High model confidence variation (std: {conf_std:.3f})")



def initialize_system(selected_models: List[str] = None):
    """Initialize the system with optional model filtering"""
    global models, label_encoder, feature_columns, crop_classes, df
    
    try:
        df = pd.read_csv(DATASET_PATH)
        
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        crop_classes = summary['dataset_info']['crops']
        feature_columns = summary['dataset_info']['features']
        
        all_models = load_all_models()
        
        # Filter models if specified
        if selected_models:
            models = {name: data for name, data in all_models.items() 
                     if name in selected_models}
            if not models:
                print(f"⚠️  No selected models found. Available models: {list(all_models.keys())}")
                print("⚠️  Using all available models instead")
                models = all_models
        else:
            models = all_models
        
        if not models:
            raise RuntimeError("No models found in models/individual directory")
        
        first_model_data = next(iter(models.values()))
        label_encoder = first_model_data.get('label_encoder')
        if label_encoder is None:
            raise RuntimeError("Label encoder not found in model data")
        
        print(f"✓ System initialized with {len(models)} model(s)")
        print(f"✓ Available crops: {len(crop_classes)}")
        print(f"✓ Using models: {list(models.keys())}")
        print(f"✓ Feature columns: {feature_columns}")
        
        # Print model types
        print("\n🤖 MODEL TYPES:")
        for model_name, model_data in models.items():
            model_type = type(model_data['model']).__name__
            print(f"  {model_name:<20}: {model_type}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize system: {str(e)}")


def test_single_case_debug(input_data: pd.DataFrame, crop: str, threshold: float) -> Dict[str, Any]:
    """
    Test a single case with detailed debugging
    """
    try:
        # Get crop index
        crop_idx = label_encoder.transform([crop])[0]
        
        # Get parameter analysis
        param_analysis = analyze_parameters(input_data, crop)
        
        # Run predictions for each model
        model_confidence = {}
        model_param_confidences = []
        model_details = {}
        
        for model_name, model_data in models.items():
            model = model_data['model']
            requires_scaling = model_data.get('requires_scaling', False)
            
            # Scale if needed
            if requires_scaling:
                scaler = model_data.get('scaler')
                input_processed = scaler.transform(input_data) if scaler else input_data.values
            else:
                input_processed = input_data.values
            
            # Get prediction probability
            try:
                proba = model.predict_proba(input_processed)[0][crop_idx]
            except Exception:
                # Fallback for models without predict_proba
                pred = model.predict(input_processed)[0]
                proba = 1.0 if pred == crop_idx else 0.0
            
            model_confidence[model_name] = float(proba)
            
            # Calculate parameter confidence
            feature_importance = model_data.get('feature_importance', [1.0] * len(feature_columns))
            param_conf = calculate_parameter_confidence(input_data, crop, feature_importance)
            model_param_confidences.append(param_conf)
            
            # Store model details
            model_details[model_name] = {
                'feature_importance': feature_importance,
                'requires_scaling': requires_scaling,
                'model_type': type(model).__name__
            }
        
        # Calculate averages
        avg_model_conf = np.mean(list(model_confidence.values())) if model_confidence else 0
        avg_param_conf = np.mean(model_param_confidences) if model_param_confidences else 0
        
        # Weighted confidence
        final_confidence = 0.8 * avg_param_conf + 0.2 * avg_model_conf
        is_suitable = final_confidence >= threshold
        
        # Get dataset stats
        crop_data = df[df['label'] == crop]
        dataset_stats = {}
        for param in feature_columns:
            param_data = crop_data[param]
            dataset_stats[param] = {
                'min': float(param_data.min()),
                'max': float(param_data.max()),
                'mean': float(param_data.mean()),
                'median': float(param_data.median()),
                'std': float(param_data.std()),
                'q1': float(param_data.quantile(0.25)),
                'q3': float(param_data.quantile(0.75)),
                'count': int(len(param_data))
            }
        
        # Calculate feature scores
        feature_scores = []
        for param in feature_columns:
            if param in param_analysis:
                feature_scores.append(param_analysis[param]['feature_score'])
        
        # Check for the paradox
        all_optimal = all(param_analysis[param]['status'] == 'optimal' for param in feature_columns)
        all_high_scores = all(score > 0.8 for score in feature_scores)
        
        return {
            'is_suitable': bool(is_suitable),
            'final_confidence': round(final_confidence, 4),
            'parameter_confidence': round(avg_param_conf, 4),
            'model_confidence_avg': round(avg_model_conf, 4),
            'ml_confidence': model_confidence,
            'crop': crop,
            'parameters_analysis': param_analysis,
            'model_details': model_details,
            'dataset_stats': dataset_stats,
            'input_values': {param: float(input_data[param].values[0]) for param in feature_columns},
            'threshold_used': threshold,
            'feature_scores': feature_scores,
            'all_parameters_optimal': all_optimal,
            'all_scores_high': all_high_scores,
            'paradox_detected': all_optimal and avg_model_conf < 0.5,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Prediction error: {str(e)}",
            "traceback": traceback.format_exc()
        }
    



def analyze_parameters(input_data: pd.DataFrame, crop: str) -> Dict[str, Dict]:
    """Analyze parameters"""
    if df is None:
        return {}
    
    crop_data = df[df['label'] == crop]
    if len(crop_data) == 0:
        return {}
    
    analysis = {}
    for param in feature_columns:
        ideal_min = crop_data[param].quantile(0.01)
        ideal_max = crop_data[param].quantile(0.99)
        current_val = input_data[param].values[0]
        
        score = feature_score(current_val, ideal_min, ideal_max)
        
        status = 'optimal'
        if current_val < ideal_min:
            status = 'low'
        elif current_val > ideal_max:
            status = 'high'
        
        analysis[param] = {
            'status': status,
            'current': float(current_val),
            'ideal_min': float(ideal_min),
            'ideal_max': float(ideal_max),
            'feature_score': float(score),
            'in_ideal_range': bool(ideal_min <= current_val <= ideal_max)
        }
    
    return analysis


def calculate_parameter_confidence(input_data: pd.DataFrame, crop: str, feature_importance: List[float]) -> float:
    """Calculate parameter confidence"""
    crop_data = df[df['label'] == crop]
    scores, weights = [], []
    
    for i, param in enumerate(feature_columns):
        min_v = crop_data[param].quantile(0.01)
        max_v = crop_data[param].quantile(0.99)
        val = input_data[param].values[0]
        
        s = feature_score(val, min_v, max_v)
        w = feature_importance[i] if i < len(feature_importance) else 1.0
        scores.append(s * w)
        weights.append(w)
    
    return sum(scores) / sum(weights) if weights else 0.0





# ------------------ Modified main function ------------------

def main():
    """
    Main function to run manual tests with enhanced analysis
    """
    print("🌾 CROP SUITABILITY DEBUGGING TOOL WITH DATASET ANALYSIS")
    print("="*60)
    
    # Show available models if MODELS_TO_USE is not set
    all_model_names = show_available_models()
    
    # Determine which models to use
    if MODELS_TO_USE is None:
        print(f"\n✓ Using ALL {len(all_model_names)} available models")
        selected_models = None
    else:
        # Filter to only include existing models
        selected_models = [m for m in MODELS_TO_USE if m in all_model_names]
        if len(selected_models) != len(MODELS_TO_USE):
            missing = set(MODELS_TO_USE) - set(selected_models)
            print(f"\n⚠️  Some specified models not found: {missing}")
        print(f"\n✓ Using {len(selected_models)} specified model(s)")
    
    # Initialize system
    try:
        initialize_system(selected_models=selected_models)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Check if crop exists
    if CROP_TO_TEST not in crop_classes:
        print(f"❌ Crop '{CROP_TO_TEST}' not found in available crops")
        print(f"Available crops: {sorted(list(crop_classes))}")
        return
    
    print(f"\n🔍 TESTING CROP: {CROP_TO_TEST}")
    print(f"   Using parameters from MANUAL_TEST_PARAMS")
    print(f"   Using {len(models)} model(s): {list(models.keys())}")
    
    # ===== NEW: Run dataset analysis =====
    print(f"\n{'='*80}")
    print("📊 RUNNING DATASET ANALYSIS...")
    print("="*80)
    
    dataset_analysis = analyze_dataset_for_crop(CROP_TO_TEST, MANUAL_TEST_PARAMS)
    print_dataset_analysis(dataset_analysis)
    
    # ===== NEW: Run model prediction analysis =====
    print(f"\n{'='*80}")
    print("🤖 ANALYZING MODEL PREDICTIONS...")
    print("="*80)
    
    prediction_analysis = analyze_model_predictions_for_crop(CROP_TO_TEST, MANUAL_TEST_PARAMS)
    print_model_prediction_analysis(prediction_analysis)
    
    # Original test (unchanged)
    print(f"\n{'='*80}")
    print("🔬 RUNNING ORIGINAL SUITABILITY TEST...")
    print("="*80)
    
    # Create input data
    input_values = []
    for param in feature_columns:
        if param in MANUAL_TEST_PARAMS:
            input_values.append(MANUAL_TEST_PARAMS[param])
        else:
            print(f"⚠️  Parameter '{param}' not in MANUAL_TEST_PARAMS, using 0")
            input_values.append(0)
    
    input_data = pd.DataFrame([input_values], columns=feature_columns)
    
    # Run test
    result = test_single_case_debug(input_data, CROP_TO_TEST, THRESHOLD)
    
    # Print results
    print_results(result)
    
    # Show how to modify settings
    print(f"\n{'='*80}")
    print("🔧 HOW TO MODIFY TEST SETTINGS:")
    print(f"  - Change CROP_TO_TEST (currently: '{CROP_TO_TEST}')")
    print(f"  - Change MANUAL_TEST_PARAMS dictionary values")
    print(f"  - Change THRESHOLD (currently: {THRESHOLD})")
    print(f"  - Change MODELS_TO_USE list (currently: {MODELS_TO_USE})")
    print(f"{'='*80}")


# ------------------ Rest of the original functions remain unchanged ------------------

# [All the previous functions remain exactly as they were: feature_score(), load_all_models(), 
# initialize_system(), analyze_parameters(), calculate_parameter_confidence(), 
# test_single_case_debug(), print_results(), show_available_models()]

if __name__ == "__main__":
    # Simply run the test with parameters defined at the top
    main()