import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load the saved models summary
print("Loading model data...")

# Method 1: Load from the all_models_artifacts.pkl file
try:
    with open('precomputation/all_models_artifacts.pkl', 'rb') as f:
        all_models_data = pickle.load(f)
    
    metrics = all_models_data['metrics']
    le = all_models_data['label_encoder']
    
    print(f"✓ Loaded data for {len(metrics)} models")
    print(f"✓ Number of classes: {len(le.classes_)}")
    
except FileNotFoundError:
    print("⚠ Could not find all_models_artifacts.pkl")
    print("Trying to load individual model files...")
    
    # Method 2: Load from individual model files
    model_files = list(Path('models/individual').glob('*.pkl'))
    metrics = {}
    
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model_name = model_data['model_name']
        metrics[model_name] = model_data['performance_metrics']
    
    # Load label encoder from any model
    if model_files:
        with open(model_files[0], 'rb') as f:
            first_model = pickle.load(f)
            le = first_model['label_encoder']
    
    print(f"✓ Loaded {len(metrics)} individual models")

# Display confusion matrices
print("\n" + "="*80)
print("CONFUSION MATRIX ANALYSIS")
print("="*80)

for model_name, model_metrics in metrics.items():
    print(f"\n{model_name.upper()}")
    print("-" * 40)
    
    # Get confusion matrix from metrics
    if 'confusion_matrix' in model_metrics:
        cm = np.array(model_metrics['confusion_matrix'])
        
        # Print basic info
        print(f"Confusion Matrix Shape: {cm.shape}")
        print(f"Total samples in test set: {cm.sum()}")
        print(f"Correct predictions: {np.trace(cm)}")
        print(f"Accuracy: {model_metrics.get('accuracy', np.trace(cm)/cm.sum()):.4f}")
        
        # Show class names if available
        if 'le' in locals() and le is not None:
            print("\nFirst 5 crop classes:")
            for i, crop in enumerate(le.classes_[:5]):
                print(f"  {i}: {crop}")
            if len(le.classes_) > 5:
                print(f"  ... (total {len(le.classes_)} crops)")
        
        # Export to CSV for detailed analysis
        if 'le' in locals() and le is not None:
            df_cm = pd.DataFrame(
                cm,
                index=[f"True_{crop}" for crop in le.classes_],
                columns=[f"Pred_{crop}" for crop in le.classes_]
            )
            
            csv_filename = f"confusion_{model_name.lower().replace(' ', '_')}.csv"
            df_cm.to_csv(csv_filename)
            print(f"✓ Confusion matrix exported to {csv_filename}")
    else:
        print("No confusion matrix data found")
    
    print("\n" + "-" * 40)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)