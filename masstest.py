import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from datetime import datetime
import sys
import os
import argparse

# Add parent directory to path to import modules if needed
sys.path.append('..')

# Load the artifacts and setup (same as in your FastAPI code)
ARTIFACTS_PATH = Path("precomputation/training_artifacts.pkl")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")

try:
    # Load training artifacts
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    
    models = artifacts['models'] 
    metrics = artifacts['metrics']
    le = artifacts['label_encoder']
    scaler = artifacts['scaler']
    X_cols = artifacts['feature_columns']
    
    # Load dataset for reference
    df = pd.read_csv(DATASET_PATH)
    
    print("✓ Successfully loaded artifacts and dataset")
    print(f"Available models: {list(models.keys())}")
    print(f"Available crops: {list(le.classes_)}")
    print(f"Feature columns: {X_cols}")
    
except FileNotFoundError as e:
    print(f"Error: Artifacts file not found: {e}")
    sys.exit(1)

def check_crop_suitability_for_row(row, model_name='XGBoost'):
    """
    Check suitability for a single row of data using XGBoost model
    
    Args:
        row: Dictionary or Series containing the data
        model_name: Name of model to use (default: XGBoost)
    
    Returns:
        Dictionary with prediction results in the same format as the API
    """
    try:
        # Convert row to dictionary if it's a Series
        if hasattr(row, 'to_dict'):
            row_dict = row.to_dict()
        else:
            row_dict = row
        
        # Extract crop from the row
        crop = row_dict.get('label')
        
        if not crop:
            return {"error": "No crop label found in row"}
        
        # Prepare input data
        input_data = pd.DataFrame([[
            row_dict.get('soil_ph', 0),
            row_dict.get('fertility_ec', 0),
            row_dict.get('humidity', 0),
            row_dict.get('sunlight', 0),
            row_dict.get('soil_temp', 0),
            row_dict.get('soil_moisture', 0)
        ]], columns=X_cols)
        
        # Check if crop exists in label encoder
        try:
            crop_idx = le.transform([crop])[0]
        except ValueError:
            return {"error": f"Crop '{crop}' not in label encoder", "crop": crop}
        
        # Get XGBoost model
        if model_name not in models:
            available = list(models.keys())
            return {"error": f"Model '{model_name}' not found. Available: {available}"}
        
        model = models[model_name]
        
        # Make prediction (XGBoost uses raw data, not scaled)
        proba = model.predict_proba(input_data)[0][crop_idx]
        
        # Determine suitability (using 70% threshold as in API)
        is_suitable = proba >= 0.65
        
        # Get ideal ranges for analysis (same as API)
        crop_data = df[df['label'] == crop]
        parameters_analysis = {}
        
        for param in X_cols:
            if len(crop_data) > 0:
                ideal_min = float(crop_data[param].quantile(0.01))
                ideal_max = float(crop_data[param].quantile(0.99))
            else:
                ideal_min = ideal_max = 0
                
            current_val = float(input_data[param].values[0])
            
            status = 'optimal'
            difference = 0
            
            if current_val < ideal_min:
                status = 'low'
                difference = current_val - ideal_min
            elif current_val > ideal_max:
                status = 'high'
                difference = current_val - ideal_max
            
            parameters_analysis[param] = {
                'status': status,
                'current': current_val,
                'ideal_min': ideal_min,
                'ideal_max': ideal_max,
                'difference': difference
            }
        
        # Return in the same format as your API
        return {
            'is_suitable': bool(is_suitable),
            'confidence': round(float(proba), 4),
            'crop': crop,
            'parameters_analysis': parameters_analysis,
            'model_used': [model_name],
            'disclaimer': "Results should be verified with local agricultural experts",
            'input_values': {
                'soil_ph': float(row_dict.get('soil_ph', 0)),
                'fertility_ec': float(row_dict.get('fertility_ec', 0)),
                'humidity': float(row_dict.get('humidity', 0)),
                'sunlight': float(row_dict.get('sunlight', 0)),
                'soil_temp': float(row_dict.get('soil_temp', 0)),
                'soil_moisture': float(row_dict.get('soil_moisture', 0))
            }
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"error": f"{str(e)}", "traceback": error_details}

def print_detailed_result(result, row_index, original_index=None):
    """
    Print detailed results for a single row in a formatted way
    """
    print("\n" + "="*70)
    if original_index is not None:
        print(f"ROW {row_index + 1} (Original index: {original_index})")
    else:
        print(f"ROW {row_index + 1}")
    print("="*70)
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
        return
    
    # Print basic info
    status = "✓ SUITABLE" if result['is_suitable'] else "✗ NOT SUITABLE"
    print(f"Crop: {result['crop']}")
    print(f"Status: {status}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {', '.join(result['model_used'])}")
    
    # Print input values
    print(f"\n📊 INPUT VALUES:")
    input_vals = result.get('input_values', {})
    for param, value in input_vals.items():
        print(f"  {param.replace('_', ' ').title():<15}: {value:.2f}")
    
    # Print parameter analysis
    print(f"\n📈 PARAMETER ANALYSIS:")
    print(f"{'Parameter':<20} {'Current':<10} {'Status':<10} {'Ideal Range':<20}")
    print(f"{'-'*60}")
    
    param_analysis = result.get('parameters_analysis', {})
    for param, analysis in param_analysis.items():
        param_name = param.replace('_', ' ').title()
        current = analysis.get('current', 0)
        status = analysis.get('status', 'unknown').upper()
        ideal_range = f"{analysis.get('ideal_min', 0):.1f}-{analysis.get('ideal_max', 0):.1f}"
        
        # Add color indicators
        status_display = status
        if status == 'LOW':
            status_display = f"🔻 {status}"
        elif status == 'HIGH':
            status_display = f"🔺 {status}"
        elif status == 'OPTIMAL':
            status_display = f"✅ {status}"
        
        print(f"{param_name:<20} {current:<10.2f} {status_display:<15} {ideal_range:<20}")
    
    print(f"\n💡 Disclaimer: {result.get('disclaimer', '')}")

def test_spc_soil_data(csv_path="dataset/SPC-soil-data.csv", start_row=0, end_row=None, sample_size=None):
    """
    Test rows in SPC-soil-data.csv using XGBoost model
    
    Args:
        csv_path: Path to SPC soil data CSV file
        start_row: Starting row index (0-based)
        end_row: Ending row index (exclusive, None for all rows)
        sample_size: Number of random rows to sample (None for sequential)
    
    Returns:
        DataFrame with test results
    """
    try:
        # Load SPC soil data
        spc_data = pd.read_csv(csv_path)
        total_rows = len(spc_data)
        print(f"✓ Loaded SPC soil data with {total_rows} rows")
        print(f"Columns: {list(spc_data.columns)}")
        
        # Validate row ranges
        if start_row < 0:
            start_row = 0
        if end_row is None:
            end_row = total_rows
        if end_row > total_rows:
            end_row = total_rows
        if start_row >= end_row:
            print(f"Error: Invalid range. start_row ({start_row}) must be less than end_row ({end_row})")
            return None, None
        
        # Select data based on parameters
        if sample_size and sample_size < (end_row - start_row):
            # Random sampling
            selected_indices = np.random.choice(range(start_row, end_row), size=sample_size, replace=False)
            selected_indices.sort()
            test_data = spc_data.iloc[selected_indices]
            selection_type = f"{sample_size} random rows"
        else:
            # Sequential range
            test_data = spc_data.iloc[start_row:end_row]
            selection_type = f"rows {start_row} to {end_row-1}"
        
        print(f"✓ Testing {len(test_data)} rows ({selection_type})")
        print(f"Model: XGBoost")
        print(f"Threshold: 70% confidence")
        
        # Test each row
        results = []
        suitable_count = 0
        unsuitable_count = 0
        error_count = 0
        
        print(f"\n{'='*60}")
        print(f"STARTING TEST BATCH")
        print(f"{'='*60}")
        
        for idx, (row_index, row) in enumerate(test_data.iterrows()):
            print(f"\n📋 Processing: Row {idx + 1}/{len(test_data)}")
            print(f"   Original CSV index: {row_index}")
            print(f"   Crop: {row['label']}")
            
            result = check_crop_suitability_for_row(row, model_name='XGBoost')
            result['original_index'] = int(row_index)
            
            # Print detailed result for this row
            print_detailed_result(result, idx, original_index=row_index)
            
            if 'error' in result:
                error_count += 1
            else:
                if result['is_suitable']:
                    suitable_count += 1
                else:
                    unsuitable_count += 1
            
            results.append(result)
        
        # Generate summary statistics
        total_tested = len(test_data)
        total_valid = suitable_count + unsuitable_count
        
        print(f"\n{'='*70}")
        print("📊 TEST BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"Rows tested: {selection_type}")
        print(f"Total predictions: {total_tested}")
        print(f"Valid predictions: {total_valid} ({total_valid/total_tested*100:.1f}%)")
        print(f"  ✅ Suitable: {suitable_count} ({suitable_count/total_tested*100:.1f}%)")
        print(f"  ❌ Unsuitable: {unsuitable_count} ({unsuitable_count/total_tested*100:.1f}%)")
        print(f"  ⚠️  Errors: {error_count} ({error_count/total_tested*100:.1f}%)")
        
        if suitable_count > 0:
            suitable_confidences = [r['confidence'] for r in results if 'error' not in r and r['is_suitable']]
            avg_confidence_suitable = np.mean(suitable_confidences) if suitable_confidences else 0
            print(f"Average confidence for suitable crops: {avg_confidence_suitable:.2%}")
        
        if unsuitable_count > 0:
            unsuitable_confidences = [r['confidence'] for r in results if 'error' not in r and not r['is_suitable']]
            avg_confidence_unsuitable = np.mean(unsuitable_confidences) if unsuitable_confidences else 0
            print(f"Average confidence for unsuitable crops: {avg_confidence_unsuitable:.2%}")
        
        # Save results to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"spc_test_results_{timestamp}.csv"
        
        # Prepare DataFrame for saving
        save_data = []
        for i, result in enumerate(results):
            if 'error' in result:
                save_data.append({
                    'original_index': result.get('original_index', 'N/A'),
                    'row_in_test': i,
                    'crop': result.get('crop', 'N/A'),
                    'status': 'ERROR',
                    'error': result['error'][:100],  # Truncate long errors
                    'confidence': 'N/A',
                    'is_suitable': 'N/A'
                })
            else:
                save_data.append({
                    'original_index': result.get('original_index', 'N/A'),
                    'row_in_test': i,
                    'crop': result['crop'],
                    'status': 'SUITABLE' if result['is_suitable'] else 'NOT_SUITABLE',
                    'error': 'N/A',
                    'confidence': result['confidence'],
                    'is_suitable': result['is_suitable']
                })
        
        results_df = pd.DataFrame(save_data)
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Save detailed results to JSON
        json_file = f"spc_test_results_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Remove traceback from results to make JSON cleaner
            clean_results = []
            for result in results:
                clean_result = result.copy()
                if 'traceback' in clean_result:
                    del clean_result['traceback']
                clean_results.append(clean_result)
            json.dump(clean_results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to: {json_file}")
        
        return results_df, results
        
    except FileNotFoundError:
        print(f"Error: SPC soil data file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error testing SPC soil data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_crop_performance(results):
    """
    Analyze performance by crop type
    
    Args:
        results: List of result dictionaries from testing
    """
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to analyze")
        return
    
    # Group by crop
    crop_stats = {}
    for result in valid_results:
        crop = result['crop']
        if crop not in crop_stats:
            crop_stats[crop] = {
                'total': 0,
                'suitable': 0,
                'unsuitable': 0,
                'confidences': []
            }
        
        crop_stats[crop]['total'] += 1
        if result['is_suitable']:
            crop_stats[crop]['suitable'] += 1
        else:
            crop_stats[crop]['unsuitable'] += 1
        crop_stats[crop]['confidences'].append(result['confidence'])
    
    # Print crop-wise statistics
    print(f"\n{'='*70}")
    print("🌱 CROP-WISE PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Crop':<20} {'Total':<8} {'Suitable':<10} {'Unsuitable':<12} {'Suitability %':<15} {'Avg Confidence':<15}")
    print(f"{'-'*80}")
    
    for crop, stats in sorted(crop_stats.items()):
        total = stats['total']
        suitable = stats['suitable']
        unsuitable = stats['unsuitable']
        suitability_pct = (suitable / total * 100) if total > 0 else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
        
        # Add emoji based on suitability percentage
        if suitability_pct >= 80:
            emoji = "🌿"
        elif suitability_pct >= 50:
            emoji = "🌱"
        else:
            emoji = "🍂"
        
        print(f"{emoji} {crop:<18} {total:<8} {suitable:<10} {unsuitable:<12} {suitability_pct:<15.1f} {avg_conf:<15.2%}")

def main():
    """
    Main function to run the SPC soil data testing with command line arguments
    """
    parser = argparse.ArgumentParser(description='Test SPC soil data with XGBoost model')
    parser.add_argument('--start', type=int, default=0, 
                       help='Starting row index (0-based, default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending row index (exclusive, default: all rows)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of random rows to sample (overrides start/end if set)')
    parser.add_argument('--csv', type=str, default="dataset/SPC-soil-data.csv",
                       help='Path to SPC soil data CSV file')
    parser.add_argument('--simple', action='store_true',
                       help='Show simplified output without detailed parameter analysis')
    
    args = parser.parse_args()
    
    print("🌾 SPC Soil Data Testing Script")
    print("=" * 50)
    
    # Run test with specified parameters
    results_df, results = test_spc_soil_data(
        csv_path=args.csv,
        start_row=args.start,
        end_row=args.end,
        sample_size=args.sample
    )
    
    if results:
        # Analyze crop performance
        analyze_crop_performance(results)
        
        # Show top 5 predictions
        print(f"\n{'='*70}")
        print("🏆 TOP 5 PREDICTIONS")
        print(f"{'='*70}")
        
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            # Sort by confidence (highest first)
            sorted_results = sorted(valid_results, key=lambda x: x['confidence'], reverse=True)
            
            for i, result in enumerate(sorted_results[:5]):
                print(f"\n#{i+1} Confidence: {result['confidence']:.2%}")
                print(f"  Crop: {result['crop']}")
                print(f"  Status: {'Suitable ✅' if result['is_suitable'] else 'Not Suitable ❌'}")
                print(f"  Original Index: {result.get('original_index', 'N/A')}")
                print("-" * 40)

def quick_test_example():
    """
    Quick test example - run this to see the format
    """
    print("QUICK TEST EXAMPLE")
    print("=" * 50)
    
    # Load a single row for testing
    spc_data = pd.read_csv("dataset/SPC-soil-data.csv")
    test_row = spc_data.iloc[0]  # First row
    
    print(f"Testing single row: {test_row['label']}")
    print(f"Row data: {dict(test_row)}")
    print("\n" + "="*50 + "\n")
    
    result = check_crop_suitability_for_row(test_row, model_name='XGBoost')
    print_detailed_result(result, 0, original_index=0)

if __name__ == "__main__":
    # To run with command line arguments, use main()
    # To run quick test example, uncomment the next line
    
    # Uncomment for quick example of output format:
    # quick_test_example()
    
    # Run main with command line arguments
    main()