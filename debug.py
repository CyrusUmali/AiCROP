import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Load artifacts and dataset
with open("precomputation/training_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

# df = pd.read_csv("dataset/augmented_crop_data.csv")
df = pd.read_csv("dataset/enhanced_crop_data.csv")
models = artifacts['models']
le = artifacts['label_encoder'] 
X_cols = artifacts['feature_columns']

# Your input values
input_values = {
     


     "soil_ph": 6.3,
  "fertility_ec": 525,
  "humidity": 74,
  "sunlight": 3200,
  "soil_temp": 29.1,
  "soil_moisture": 88.0,
}

crop_name = "Ampalaya"

print("="*80)
print(f"DEBUGGING SUITABILITY CHECK FOR: {crop_name}")
print("="*80)

# 1. Check crop data distribution
crop_data = df[df['label'] == crop_name]
print(f"\n1. CROP DATA IN DATASET")
print(f"   Total samples: {len(crop_data)}")

if len(crop_data) == 0:
    print(f"   ⚠️ ERROR: No data found for '{crop_name}'")
    print(f"   Available crops: {sorted(df['label'].unique())}")
else:
    print(f"\n2. PARAMETER STATISTICS FOR {crop_name}")
    print("-"*80)
    for param in X_cols:
        stats = crop_data[param].describe()
        input_val = input_values[param]
        
        print(f"\n{param.upper()}:")
        print(f"   Input value:  {input_val:.2f}")
        print(f"   Mean:         {stats['mean']:.2f}")
        print(f"   Std Dev:      {stats['std']:.2f}")
        print(f"   Min:          {stats['min']:.2f}")
        print(f"   25th %ile:    {stats['25%']:.2f}")
        print(f"   Median:       {stats['50%']:.2f}")
        print(f"   75th %ile:    {stats['75%']:.2f}")
        print(f"   Max:          {stats['max']:.2f}")
        
        # Calculate z-score
        z_score = (input_val - stats['mean']) / stats['std']
        print(f"   Z-score:      {z_score:.2f} {'⚠️ OUTLIER' if abs(z_score) > 2 else '✓'}")

    # 3. Check model predictions in detail
    print("\n3. MODEL PREDICTION ANALYSIS")
    print("-"*80)
    
    input_df = pd.DataFrame([list(input_values.values())], columns=X_cols)
    
    xgb_model = models['XGBoost']
    probabilities = xgb_model.predict_proba(input_df)[0]
    
    # Get top 5 predictions
    top_indices = np.argsort(probabilities)[::-1][:5]
    
    print("\nTop 5 Predicted Crops:")
    for i, idx in enumerate(top_indices, 1):
        crop = le.classes_[idx]
        prob = probabilities[idx]
        print(f"   {i}. {crop:20s} - {prob*100:5.2f}%")
    
    # Find Ampalaya's ranking
    ampalaya_idx = le.transform([crop_name])[0]
    ampalaya_prob = probabilities[ampalaya_idx]
    ampalaya_rank = np.where(np.argsort(probabilities)[::-1] == ampalaya_idx)[0][0] + 1
    
    print(f"\n{crop_name}:")
    print(f"   Probability:  {ampalaya_prob*100:.2f}%")
    print(f"   Rank:         {ampalaya_rank} out of {len(le.classes_)}")

    # 4. Find similar samples in training data
    print(f"\n4. SIMILAR SAMPLES IN TRAINING DATA")
    print("-"*80)
    
    # Calculate distance to each Ampalaya sample
    distances = []
    for _, row in crop_data.iterrows():
        dist = np.sqrt(sum((row[param] - input_values[param])**2 for param in X_cols))
        distances.append(dist)
    
    if distances:
        min_dist = min(distances)
        avg_dist = np.mean(distances)
        print(f"   Minimum distance to Ampalaya sample: {min_dist:.2f}")
        print(f"   Average distance to Ampalaya samples: {avg_dist:.2f}")
        
        # Find the closest sample
        closest_idx = np.argmin(distances)
        closest_sample = crop_data.iloc[closest_idx]
        
        print(f"\n   Closest Ampalaya sample:")
        for param in X_cols:
            diff = input_values[param] - closest_sample[param]
            print(f"      {param:15s}: {closest_sample[param]:8.2f} (diff: {diff:+7.2f})")

    # 5. Check feature importance
    print(f"\n5. FEATURE IMPORTANCE (XGBoost)")
    print("-"*80)
    
    importances = xgb_model.feature_importances_
    feature_importance = sorted(zip(X_cols, importances), key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance:
        print(f"   {feature:15s}: {importance:.4f}")

print("\n" + "="*80)
print("DEBUGGING COMPLETE")
print("="*80)