import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse
from typing import List, Dict, Any, Optional

# ------------------ Setup and Initialization ------------------

# Paths (relative to script location)
MODELS_DIR = Path("models/individual")
DATASET_PATH = Path("dataset/enhanced_crop_data.csv")
SUMMARY_PATH = Path("models/model_summary.json")

# Global variables
models: Dict[str, dict] = {}
label_encoder = None
feature_columns = None
crop_classes = None
df = None


def feature_score(value, min_val, max_val):
    """Calculate feature score (same as in API)"""
    if min_val <= value <= max_val:
        return 1.0
    range_width = max_val - min_val if max_val != min_val else 1
    if value < min_val:
        return max(0.0, 1 - abs(value - min_val) / range_width)
    else:
        return max(0.0, 1 - abs(value - max_val) / range_width)


def load_model(model_name: str) -> dict:
    """Load individual model"""
    model_filename = model_name.lower().replace(' ', '_') + '.pkl'
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


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


def initialize_system():
    """Initialize the system (same as in API)"""
    global models, label_encoder, feature_columns, crop_classes, df
    
    try:
        df = pd.read_csv(DATASET_PATH)
        
        with open(SUMMARY_PATH, 'r') as f:
            summary = json.load(f)
        
        crop_classes = summary['dataset_info']['crops']
        feature_columns = summary['dataset_info']['features']
        
        models = load_all_models()
        if not models:
            raise RuntimeError("No models found in models/individual directory")
        
        first_model_data = next(iter(models.values()))
        label_encoder = first_model_data.get('label_encoder')
        if label_encoder is None:
            raise RuntimeError("Label encoder not found in model data")
        
        print(f"✓ System initialized with {len(models)} models")
        print(f"✓ Available crops: {len(crop_classes)}")
        print(f"✓ Available models: {list(models.keys())}")
        print(f"✓ Feature columns: {feature_columns}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize system: {str(e)}")


def analyze_parameters(input_data: pd.DataFrame, crop: str) -> Dict[str, Dict]:
    """Analyze parameters (adapted from API)"""
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
        
        status = 'optimal'
        difference = 0
        
        if current_val < ideal_min:
            status = 'low'
            difference = current_val - ideal_min
        elif current_val > ideal_max:
            status = 'high'
            difference = current_val - ideal_max
        
        analysis[param] = {
            'status': status,
            'current': float(current_val),
            'ideal_min': float(ideal_min),
            'ideal_max': float(ideal_max),
            'difference': float(difference)
        }
    
    return analysis


def calculate_parameter_confidence(input_data: pd.DataFrame, crop: str, feature_importance: List[float]) -> float:
    """Calculate parameter confidence (same as in API)"""
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


# ------------------ Core Testing Function ------------------

def check_crop_suitability_for_row(
    row: pd.Series,
    selected_models: Optional[List[str]] = None,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Check crop suitability for a single row (replicates API endpoint logic)
    
    Args:
        row: DataFrame row containing data
        selected_models: List of model names to use (None for all)
        threshold: Confidence threshold for suitability
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert row to dictionary
        row_dict = row.to_dict()
        
        # Extract parameters
        soil_ph = row_dict.get('soil_ph')
        fertility_ec = row_dict.get('fertility_ec')
        humidity = row_dict.get('humidity')
        sunlight = row_dict.get('sunlight')
        soil_temp = row_dict.get('soil_temp')
        soil_moisture = row_dict.get('soil_moisture')
        crop = row_dict.get('label')  # Assuming 'label' column contains crop name
        
        # Validate required parameters
        missing_params = []
        for param in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']:
            if param not in row_dict:
                missing_params.append(param)
        
        if missing_params:
            return {"error": f"Missing parameters: {missing_params}"}
        
        if not crop:
            return {"error": "No crop label found in row"}
        
        # Check if crop exists
        if crop not in crop_classes:
            return {"error": f"Crop '{crop}' not found in available crops"}
        
        # Prepare input data
        input_data = pd.DataFrame([[
            soil_ph, fertility_ec, humidity,
            sunlight, soil_temp, soil_moisture
        ]], columns=feature_columns)
        
        # Get crop index
        crop_idx = label_encoder.transform([crop])[0]
        
        # Determine which models to use
        models_to_use = selected_models if selected_models else list(models.keys())
        valid_models = [m for m in models_to_use if m in models]
        
        if not valid_models:
            return {"error": f"No valid models selected. Available: {list(models.keys())}"}
        
        # Run predictions for each model
        model_confidence = {}
        model_param_confidences = []
        
        for model_name in valid_models:
            model_data = models[model_name]
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
        
        # Calculate averages
        avg_model_conf = np.mean(list(model_confidence.values())) if model_confidence else 0
        avg_param_conf = np.mean(model_param_confidences) if model_param_confidences else 0
        
        
        final_confidence = 0.8 * avg_param_conf + 0.2 * avg_model_conf

        is_suitable = final_confidence >= threshold
        
        # Get parameter analysis
        param_analysis = analyze_parameters(input_data, crop)
        
        return {
            'is_suitable': bool(is_suitable),
            'final_confidence': round(final_confidence, 4),
            'parameter_confidence': round(avg_param_conf, 4),
            'ml_confidence': model_confidence,
            'crop': crop,
            'parameters_analysis': param_analysis,
            'model_used': valid_models,
            'input_values': {
                'soil_ph': float(soil_ph),
                'fertility_ec': float(fertility_ec),
                'humidity': float(humidity),
                'sunlight': float(sunlight),
                'soil_temp': float(soil_temp),
                'soil_moisture': float(soil_moisture)
            },
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Prediction error: {str(e)}",
            "traceback": traceback.format_exc(),
            "row_data": dict(row)
        }


# ------------------ Output and Visualization ------------------

def print_detailed_result(result: Dict, row_index: int, original_index: Optional[int] = None):
    """
    Print detailed results for a single row
    """
    print("\n" + "="*70)
    if original_index is not None:
        print(f"ROW {row_index + 1} (Original index: {original_index})")
    else:
        print(f"ROW {row_index + 1}")
    print("="*70)
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
        if 'traceback' in result:
            print(f"Traceback (first 200 chars): {result['traceback'][:200]}...")
        return
    
    # Basic info
    status = "✅ SUITABLE" if result['is_suitable'] else "❌ NOT SUITABLE"
    print(f"Crop: {result['crop']}")
    print(f"Status: {status}")
    print(f"Final Confidence: {result['final_confidence']:.2%}")
    print(f"Parameter Confidence: {result['parameter_confidence']:.2%}")
    print(f"ML Confidence: {result['parameter_confidence']:.2%}")
    print(f"Models Used: {', '.join(result['model_used'])}")
    print(f"Threshold: {result.get('threshold_used', 0.7):.0%}")
    
    # ML model confidences
    print(f"\n🤖 MODEL CONFIDENCES:")
    for model_name, confidence in result['ml_confidence'].items():
        print(f"  {model_name:<15}: {confidence:.2%}")
    
    # Input values
    print(f"\n📊 INPUT VALUES:")
    input_vals = result.get('input_values', {})
    for param, value in input_vals.items():
        print(f"  {param.replace('_', ' ').title():<15}: {value:.2f}")
    
    # Parameter analysis
    print(f"\n📈 PARAMETER ANALYSIS:")
    print(f"{'Parameter':<20} {'Current':<10} {'Status':<10} {'Ideal Range':<20} {'Score':<10}")
    print(f"{'-'*80}")
    
    param_analysis = result.get('parameters_analysis', {})
    for param, analysis in param_analysis.items():
        param_name = param.replace('_', ' ').title()
        current = analysis.get('current', 0)
        status = analysis.get('status', 'unknown').upper()
        ideal_min = analysis.get('ideal_min', 0)
        ideal_max = analysis.get('ideal_max', 0)
        ideal_range = f"{ideal_min:.1f}-{ideal_max:.1f}"
        
        # Calculate score for this parameter
        score = feature_score(current, ideal_min, ideal_max)
        
        # Status indicators
        if status == 'LOW':
            status_display = f"🔻 {status}"
        elif status == 'HIGH':
            status_display = f"🔺 {status}"
        elif status == 'OPTIMAL':
            status_display = f"✅ {status}"
        else:
            status_display = status
        
        print(f"{param_name:<20} {current:<10.2f} {status_display:<15} {ideal_range:<20} {score:<10.2f}")


# ------------------ Batch Testing ------------------

def test_batch_data(
    csv_path: str,
    start_row: int = 0,
    end_row: Optional[int] = None,
    sample_size: Optional[int] = None,
    selected_models: Optional[List[str]] = None,
    threshold: float = 0.7,
    output_dir: str = "test_results"
) -> tuple:
    """
    Test multiple rows from a CSV file
    
    Args:
        csv_path: Path to CSV file
        start_row: Starting row index (0-based)
        end_row: Ending row index (None for all)
        sample_size: Number of random rows to sample
        selected_models: List of model names to use
        threshold: Confidence threshold
        output_dir: Directory to save results
        
    Returns:
        Tuple of (results_df, detailed_results)
    """
    try:
        # Load test data
        test_data = pd.read_csv(csv_path)
        total_rows = len(test_data)
        print(f"✓ Loaded test data from {csv_path}")
        print(f"✓ Total rows: {total_rows}")
        print(f"✓ Columns: {list(test_data.columns)}")
        
        # Validate row ranges
        if end_row is None:
            end_row = total_rows
        if start_row < 0:
            start_row = 0
        if end_row > total_rows:
            end_row = total_rows
        if start_row >= end_row:
            print(f"⚠ Warning: Invalid range. Using all rows.")
            start_row, end_row = 0, total_rows
        
        # Select data
        if sample_size and sample_size < (end_row - start_row):
            # Random sampling
            selected_indices = np.random.choice(range(start_row, end_row), size=sample_size, replace=False)
            selected_indices.sort()
            batch_data = test_data.iloc[selected_indices]
            selection_type = f"{sample_size} random rows"
        else:
            # Sequential range
            batch_data = test_data.iloc[start_row:end_row]
            selection_type = f"rows {start_row} to {end_row-1}"
        
        print(f"\n🚀 TESTING PARAMETERS")
        print(f"   Selection: {selection_type}")
        print(f"   Models: {selected_models if selected_models else 'All available'}")
        print(f"   Threshold: {threshold:.0%}")
        print(f"   Total rows to test: {len(batch_data)}")
        
        # Run tests
        results = []
        suitable_count = 0
        unsuitable_count = 0
        error_count = 0
        
        print(f"\n{'='*60}")
        print(f"STARTING TEST BATCH")
        print(f"{'='*60}")
        
        for idx, (row_index, row) in enumerate(batch_data.iterrows()):
            print(f"\n📋 Processing: Row {idx + 1}/{len(batch_data)}")
            print(f"   Original index: {row_index}")
            print(f"   Crop: {row.get('label', 'Unknown')}")
            
            result = check_crop_suitability_for_row(
                row, 
                selected_models=selected_models,
                threshold=threshold
            )
            result['original_index'] = int(row_index)
            result['row_in_batch'] = idx
            
            # Print detailed result
            print_detailed_result(result, idx, original_index=row_index)
            
            # Count results
            if 'error' in result:
                error_count += 1
            else:
                if result['is_suitable']:
                    suitable_count += 1
                else:
                    unsuitable_count += 1
            
            results.append(result)
        
        # Generate summary
        total_tested = len(batch_data)
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
            suitable_confidences = [r['final_confidence'] for r in results if 'error' not in r and r['is_suitable']]
            avg_confidence_suitable = np.mean(suitable_confidences) if suitable_confidences else 0
            print(f"Average confidence for suitable crops: {avg_confidence_suitable:.2%}")
        
        if unsuitable_count > 0:
            unsuitable_confidences = [r['final_confidence'] for r in results if 'error' not in r and not r['is_suitable']]
            avg_confidence_unsuitable = np.mean(unsuitable_confidences) if unsuitable_confidences else 0
            print(f"Average confidence for unsuitable crops: {avg_confidence_unsuitable:.2%}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary CSV
        summary_data = []
        for result in results:
            if 'error' in result:
                summary_data.append({
                    'original_index': result.get('original_index', 'N/A'),
                    'row_in_batch': result.get('row_in_batch', 'N/A'),
                    'crop': result.get('crop', 'N/A'),
                    'status': 'ERROR',
                    'error': result['error'][:100],
                    'final_confidence': 'N/A',
                    'parameter_confidence': 'N/A',
                    'is_suitable': 'N/A',
                    'models_used': 'N/A'
                })
            else:
                summary_data.append({
                    'original_index': result.get('original_index', 'N/A'),
                    'row_in_batch': result.get('row_in_batch', 'N/A'),
                    'crop': result['crop'],
                    'status': 'SUITABLE' if result['is_suitable'] else 'NOT_SUITABLE',
                    'error': 'N/A',
                    'final_confidence': result['final_confidence'],
                    'parameter_confidence': result['parameter_confidence'],
                    'is_suitable': result['is_suitable'],
                    'models_used': ', '.join(result['model_used'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = os.path.join(output_dir, f"batch_test_summary_{timestamp}.csv")
        summary_df.to_csv(csv_file, index=False)
        print(f"\n💾 Summary saved to: {csv_file}")
        
        # Save detailed results to JSON
        json_file = os.path.join(output_dir, f"batch_test_detailed_{timestamp}.json")
        with open(json_file, 'w') as f:
            # Clean up results for JSON serialization
            clean_results = []
            for result in results:
                clean_result = result.copy()
                if 'traceback' in clean_result:
                    del clean_result['traceback']
                clean_results.append(clean_result)
            json.dump(clean_results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to: {json_file}")
        
        return summary_df, results
        
    except FileNotFoundError:
        print(f"❌ Error: Test data file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"❌ Error testing batch data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_crop_performance(results: List[Dict]):
    """
    Analyze performance by crop type
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
                'final_confidences': [],
                'parameter_confidences': []
            }
        
        crop_stats[crop]['total'] += 1
        if result['is_suitable']:
            crop_stats[crop]['suitable'] += 1
        else:
            crop_stats[crop]['unsuitable'] += 1
        
        crop_stats[crop]['final_confidences'].append(result['final_confidence'])
        crop_stats[crop]['parameter_confidences'].append(result['parameter_confidence'])
    
    # Print analysis
    print(f"\n{'='*80}")
    print("🌱 CROP-WISE PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Crop':<20} {'Total':<8} {'Suitable':<10} {'% Suitable':<12} {'Avg Final Conf':<15} {'Avg Param Conf':<15}")
    print(f"{'-'*80}")
    
    for crop, stats in sorted(crop_stats.items()):
        total = stats['total']
        suitable = stats['suitable']
        suitability_pct = (suitable / total * 100) if total > 0 else 0
        avg_final_conf = np.mean(stats['final_confidences']) if stats['final_confidences'] else 0
        avg_param_conf = np.mean(stats['parameter_confidences']) if stats['parameter_confidences'] else 0
        
        # Emoji indicators
        if suitability_pct >= 80:
            emoji = "🌿🌿"
        elif suitability_pct >= 50:
            emoji = "🌿"
        else:
            emoji = "🍂"
        
        print(f"{emoji} {crop:<18} {total:<8} {suitable:<10} {suitability_pct:<12.1f}% {avg_final_conf:<15.2%} {avg_param_conf:<15.2%}")


# ------------------ Main Function ------------------

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Batch test crop suitability model')
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with test data')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting row index (0-based, default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending row index (exclusive, default: all rows)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of random rows to sample')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to use (space-separated, default: all)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for suitability (default: 0.7)')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output for each row')
    
    args = parser.parse_args()
    
    print("🌾 CROP SUITABILITY BATCH TESTING SCRIPT")
    print("=" * 60)
    
    # Initialize system
    try:
        initialize_system()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Run batch test
    summary_df, results = test_batch_data(
        csv_path=args.csv,
        start_row=args.start,
        end_row=args.end,
        sample_size=args.sample,
        selected_models=args.models,
        threshold=args.threshold,
        output_dir=args.output
    )
    
    if results:
        # Analyze crop performance
        analyze_crop_performance(results)
        
        # Show top predictions
        print(f"\n{'='*70}")
        print("🏆 TOP PREDICTIONS")
        print(f"{'='*70}")
        
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            # Sort by final confidence
            sorted_results = sorted(valid_results, key=lambda x: x['final_confidence'], reverse=True)
            
            for i, result in enumerate(sorted_results[:5]):
                print(f"\n#{i+1} Final Confidence: {result['final_confidence']:.2%}")
                print(f"   Crop: {result['crop']}")
                print(f"   Status: {'✅ Suitable' if result['is_suitable'] else '❌ Not Suitable'}")
                print(f"   Models: {', '.join(result['model_used'])}")
                print(f"   Original Index: {result.get('original_index', 'N/A')}")
                print("-" * 40)


def quick_test():
    """Quick test with a few rows"""
    print("QUICK TEST")
    print("=" * 50)
    
    # Initialize
    try:
        initialize_system()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Create a test row (or load from dataset)
    if df is not None:
        test_row = df.iloc[0]
        print(f"Testing with first row from dataset:")
        print(f"Crop: {test_row['label']}")
        print(f"Parameters: {dict(test_row[feature_columns])}")
        
        result = check_crop_suitability_for_row(test_row)
        print_detailed_result(result, 0)
    else:
        print("No dataset loaded")


if __name__ == "__main__":
    # For quick testing
    # quick_test()
    
    # For batch testing with command line
    main()