import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (update if needed)
ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")

# Load artifacts
print("Loading artifacts...")
with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

models = artifacts['models']
le = artifacts['label_encoder']
scaler = artifacts['scaler']
X_cols = artifacts['feature_columns']

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Your test input
test_input = {
    "soil_ph": 6.5,
    "fertility_ec": 475.0,
    "humidity": 80.0,
    "sunlight": 1215.0,
    "soil_temp": 27.0,
    "soil_moisture": 99.0,
    "crop": "Coconut"
}

def debug_confidence_issue(test_data, crop_name):
    """Debug why confidence is low despite optimal parameters"""
    print("=" * 80)
    print(f"DEBUGGING CONFIDENCE ISSUE FOR: {crop_name}")
    print("=" * 80)
    
    # 1. Check if crop exists in dataset
    print("\n1. CROP DATA ANALYSIS:")
    print("-" * 40)
    crop_data = df[df['label'] == crop_name]
    print(f"Total samples for {crop_name}: {len(crop_data)}")
    
    if len(crop_data) == 0:
        print(f"ERROR: No data found for '{crop_name}' in dataset!")
        return
    
    # 2. Create input DataFrame
    input_df = pd.DataFrame([[
        test_data['soil_ph'],
        test_data['fertility_ec'],
        test_data['humidity'],
        test_data['sunlight'],
        test_data['soil_temp'],
        test_data['soil_moisture']
    ]], columns=X_cols)
    
    # 3. Analyze parameters vs dataset
    print("\n2. PARAMETER COMPARISON WITH DATASET:")
    print("-" * 40)
    
    for param in X_cols:
        crop_param_data = crop_data[param]
        test_value = test_data[param]
        
        mean_val = crop_param_data.mean()
        std_val = crop_param_data.std()
        min_val = crop_param_data.min()
        max_val = crop_param_data.max()
        q1 = crop_param_data.quantile(0.25)
        q3 = crop_param_data.quantile(0.75)
        
        # Calculate z-score (how many standard deviations from mean)
        z_score = (test_value - mean_val) / std_val if std_val > 0 else 0
        
        print(f"\n{param}:")
        print(f"  Test value: {test_value}")
        print(f"  Dataset range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"  Mean ± Std: {mean_val:.2f} ± {std_val:.2f}")
        print(f"  IQR (25-75%): [{q1:.2f}, {q3:.2f}]")
        print(f"  Z-score: {z_score:.2f} (ideal: -1 to +1)")
        
        if abs(z_score) > 2:
            print(f"  ⚠️  WARNING: Value is {abs(z_score):.1f} std deviations from mean!")
    
    # 4. Get model predictions
    print("\n3. MODEL PREDICTIONS:")
    print("-" * 40)
    
    # Get crop index
    try:
        crop_idx = le.transform([crop_name])[0]
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Available crops: {list(le.classes_)}")
        return
    
    predictions = {}
    
    for model_name, model in models.items():
        # Prepare input
        if model_name == 'Logistic Regression':
            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0][crop_idx]
        else:
            proba = model.predict_proba(input_df)[0][crop_idx]
        
        predictions[model_name] = proba
        
        print(f"{model_name}:")
        print(f"  Probability for {crop_name}: {proba:.6f}")
        
        # Get top 3 predictions for context
        if model_name == 'Logistic Regression':
            all_proba = model.predict_proba(input_scaled)[0]
        else:
            all_proba = model.predict_proba(input_df)[0]
        
        top_3_idx = np.argsort(all_proba)[-3:][::-1]
        top_3_crops = le.inverse_transform(top_3_idx)
        top_3_proba = all_proba[top_3_idx]
        
        print(f"  Top 3 predictions:")
        for crop, prob in zip(top_3_crops, top_3_proba):
            print(f"    - {crop}: {prob:.4f}")
    
    # 5. Check closest samples in dataset
    print("\n4. FINDING CLOSEST DATASET SAMPLES:")
    print("-" * 40)
    
    # Calculate Euclidean distance to all crop samples
    crop_features = crop_data[X_cols].values
    test_features = input_df.values
    
    distances = []
    for i, sample in enumerate(crop_features):
        dist = np.linalg.norm(sample - test_features)
        distances.append((i, dist, sample))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    print(f"Top 5 closest samples in dataset:")
    for i, (idx, dist, features) in enumerate(distances[:5]):
        print(f"\nSample {i+1} (distance: {dist:.2f}):")
        for param, value in zip(X_cols, features):
            print(f"  {param}: {value:.2f}")
    
    # 6. Feature importance analysis (for tree-based models)
    print("\n5. FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    
    # Check XGBoost feature importance
    if 'XGBoost' in models:
        xgb_model = models['XGBoost']
        feature_importance = xgb_model.feature_importances_
        
        print("XGBoost Feature Importance:")
        for param, importance in zip(X_cols, feature_importance):
            print(f"  {param}: {importance:.4f}")
        
        # Identify which features might be causing low confidence
        print("\nFeature values vs importance:")
        for param in X_cols:
            importance = feature_importance[list(X_cols).index(param)]
            test_val = test_data[param]
            crop_mean = crop_data[param].mean()
            diff = test_val - crop_mean
            
            print(f"  {param}:")
            print(f"    Importance: {importance:.4f}")
            print(f"    Test value: {test_val:.2f}")
            print(f"    Dataset mean: {crop_mean:.2f}")
            print(f"    Difference: {diff:.2f}")
            
            if importance > 0.15 and abs(diff) > crop_data[param].std():
                print(f"    ⚠️  High importance with significant deviation!")
    
    # 7. Check class distribution
    print("\n6. CLASS DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    total_samples = len(df)
    crop_samples = len(crop_data)
    crop_percentage = (crop_samples / total_samples) * 100
    
    print(f"Total dataset samples: {total_samples}")
    print(f"{crop_name} samples: {crop_samples} ({crop_percentage:.1f}%)")
    
    # Check if class is imbalanced
    if crop_percentage < 5:
        print(f"⚠️  {crop_name} is a minority class ({crop_percentage:.1f}%)")
        print("   Models might be biased against minority classes")
    
    # 8. Statistical analysis
    print("\n7. STATISTICAL OUTLIER DETECTION:")
    print("-" * 40)
    
    for param in X_cols:
        test_val = test_data[param]
        param_data = crop_data[param]
        
        # Calculate percentiles
        p5 = param_data.quantile(0.05)
        p95 = param_data.quantile(0.95)
        
        if test_val < p5 or test_val > p95:
            percentile = np.sum(param_data <= test_val) / len(param_data) * 100
            print(f"⚠️  {param}: Test value ({test_val}) is at {percentile:.1f}th percentile")
            print(f"    5th-95th percentile range: [{p5:.2f}, {p95:.2f}]")

def visualize_parameters(test_data, crop_name):
    """Create visualizations to understand parameter distributions"""
    crop_data = df[df['label'] == crop_name]
    
    if len(crop_data) == 0:
        print(f"No data to visualize for {crop_name}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, param in enumerate(X_cols):
        ax = axes[idx]
        
        # Plot distribution
        sns.histplot(crop_data[param], ax=ax, kde=True, color='skyblue', alpha=0.6)
        
        # Add test value line
        test_val = test_data[param]
        ax.axvline(x=test_val, color='red', linestyle='--', linewidth=2, label='Test Value')
        
        # Add mean line
        mean_val = crop_data[param].mean()
        ax.axvline(x=mean_val, color='green', linestyle='-', linewidth=1, label='Mean')
        
        # Add IQR
        q1 = crop_data[param].quantile(0.25)
        q3 = crop_data[param].quantile(0.75)
        ax.axvspan(q1, q3, alpha=0.2, color='yellow', label='IQR (25-75%)')
        
        ax.set_title(f"{param}\nTest: {test_val:.1f}, Mean: {mean_val:.1f}")
        ax.set_xlabel(param)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.suptitle(f"Parameter Distribution for {crop_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Run the debug analysis
debug_confidence_issue(test_input, test_input['crop'])

# Optional: Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)
visualize_parameters(test_input, test_input['crop'])

# Additional analysis: Check all crops
print("\n" + "=" * 80)
print("COMPARING WITH OTHER CROPS:")
print("=" * 80)

# Get predictions for all crops with XGBoost
xgb_model = models['XGBoost']
input_df = pd.DataFrame([[
    test_input['soil_ph'],
    test_input['fertility_ec'],
    test_input['humidity'],
    test_input['sunlight'],
    test_input['soil_temp'],
    test_input['soil_moisture']
]], columns=X_cols)

all_proba = xgb_model.predict_proba(input_df)[0]
sorted_idx = np.argsort(all_proba)[::-1]

print("\nTop 10 predicted crops for these conditions:")
print("-" * 50)
for i, idx in enumerate(sorted_idx[:10]):
    crop = le.inverse_transform([idx])[0]
    prob = all_proba[idx]
    print(f"{i+1}. {crop}: {prob:.4f}")
    
    if crop == test_input['crop']:
        print(f"   ← Your selected crop (rank: {i+1})")