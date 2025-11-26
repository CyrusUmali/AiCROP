# show_crops.py
import pickle
import pandas as pd
from pathlib import Path

def show_all_crops():
    """Show all available crops in the system"""
    
    ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
    DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
    
    print("🌱 ALL AVAILABLE CROPS IN THE SYSTEM")
    print("=" * 50)
    
    try:
        # Load training artifacts
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = pickle.load(f)
        
        le = artifacts['label_encoder']
        models = artifacts['models']
        
        # Get crops from label encoder
        available_crops = le.classes_
        
        print("📊 FROM LABEL ENCODER (What the API accepts):")
        print("-" * 40)
        for i, crop in enumerate(sorted(available_crops), 1):
            print(f"  {i:2d}. {crop}")
        print(f"\nTotal: {len(available_crops)} crops")
        
    except FileNotFoundError:
        print("❌ Artifacts file not found! Run training first.")
        available_crops = []
    
    try:
        # Load dataset to show additional info
        df = pd.read_csv(DATASET_PATH)
        
        if 'label' in df.columns:
            dataset_crops = sorted(df['label'].unique())
            
            print(f"\n📁 FROM DATASET ({len(dataset_crops)} crops):")
            print("-" * 40)
            for i, crop in enumerate(dataset_crops, 1):
                print(f"  {i:2d}. {crop}")
                
            # Show sample counts
            print(f"\n📈 SAMPLE COUNTS PER CROP:")
            print("-" * 40)
            crop_counts = df['label'].value_counts()
            for crop in sorted(dataset_crops):
                count = crop_counts.get(crop, 0)
                print(f"  {crop:15}: {count:3} samples")
                
    except FileNotFoundError:
        print("❌ Dataset file not found!")
    
    print(f"\n💡 Available ML Models: {list(models.keys())}")
    print(f"\n🎯 Use these exact crop names in your API requests!")
    print("   (case-insensitive, but use lowercase for consistency)")

if __name__ == "__main__":
    show_all_crops()