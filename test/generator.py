import pandas as pd
import numpy as np
import os
from scipy import stats

class SmartCropDataGenerator:
    def __init__(self, existing_csv_path=None, realism_mode=True):
        """
        Initialize the crop data generator that learns from existing data
        
        Parameters:
        existing_csv_path (str): Path to the existing CSV file (optional)
        realism_mode (bool): If True, uses correlations from real data
        """
        self.existing_csv_path = existing_csv_path
        self.realism_mode = realism_mode
        self.existing_data = pd.DataFrame(
            columns=['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
        )
        self.crop_profiles = {}
        
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
            self.learn_crop_profiles()
    
    def load_existing_data(self):
        """Load existing crop data"""
        try:
            existing = pd.read_csv(self.existing_csv_path)
            if not existing.empty:
                expected_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
                if set(existing.columns) == set(expected_cols):
                    numeric_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
                    
                    original_data = existing.copy()
                    
                    for col in numeric_cols:
                        existing[col] = pd.to_numeric(existing[col], errors='coerce')
                    
                    invalid_rows = existing[existing[numeric_cols].isna().any(axis=1)]
                    
                    if len(invalid_rows) > 0:
                        print(f"\n⚠ Found {len(invalid_rows)} rows with invalid data:")
                        print("="*80)
                        
                        for idx in invalid_rows.index:
                            original_row = original_data.loc[idx]
                            converted_row = existing.loc[idx]
                            
                            print(f"\nRow {idx + 2} (CSV line): {original_row['label']}")
                            
                            problems = []
                            for col in numeric_cols:
                                original_val = original_row[col]
                                converted_val = converted_row[col]
                                
                                if pd.isna(converted_val):
                                    problems.append(f"  ✗ {col}: '{original_val}' (can't convert to number)")
                            
                            for problem in problems:
                                print(problem)
                        
                        print("="*80)
                    
                    existing = existing.dropna()
                    existing['label'] = existing['label'].str.strip()
                    
                    self.existing_data = pd.concat([self.existing_data, existing], ignore_index=True)
                    print(f"✓ Loaded existing data: {len(existing)} valid rows")
                    print(f"✓ Existing crops: {list(existing['label'].unique())}")
                else:
                    print("⚠ Warning: Existing CSV has different columns. Starting with empty dataset.")
                    print(f"   Expected: {expected_cols}")
                    print(f"   Found: {list(existing.columns)}")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            import traceback
            traceback.print_exc()
    
    def learn_crop_profiles(self):
        """Learn statistical profiles from existing data for each crop"""
        if len(self.existing_data) == 0:
            print("No existing data to learn from")
            return
        
        feature_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
        
        # FIXED: Define minimum realistic standard deviations per feature
        min_stds = {
            'soil_ph': 0.15,           # At least 0.15 pH units variation
            'fertility_ec': 20,         # At least 20 units
            'humidity': 3.0,            # At least 3% variation
            'sunlight': 200,            # At least 200 lux variation
            'soil_temp': 0.5,           # At least 0.5°C variation
            'soil_moisture': 3.0        # At least 3% variation
        }
        
        for crop in self.existing_data['label'].unique():
            crop_data = self.existing_data[self.existing_data['label'] == crop]
            n_samples = len(crop_data)
            
            if n_samples < 2:
                print(f"⚠ Skipping {crop}: only {n_samples} sample (need 2+ for learning)")
                continue
            
            profile = {
                'means': {},
                'stds': {},
                'ranges': {},
                'correlations': None,
                'n_samples': n_samples
            }
            
            for col in feature_cols:
                profile['means'][col] = crop_data[col].mean()
                
                # FIXED: Use realistic minimum std OR 15% of range, whichever is larger
                observed_std = crop_data[col].std()
                range_span = crop_data[col].max() - crop_data[col].min()
                range_based_std = range_span * 0.15  # 15% of observed range
                
                # Use the maximum of: observed std, minimum realistic std, or 15% of range
                profile['stds'][col] = max(observed_std, min_stds[col], range_based_std)
                
                # FIXED: Expand ranges by 10% on each side for realistic variation
                min_val = crop_data[col].min()
                max_val = crop_data[col].max()
                buffer = (max_val - min_val) * 0.1
                profile['ranges'][col] = (max(0, min_val - buffer), max_val + buffer)
            
            if self.realism_mode and n_samples >= 5:
                profile['correlations'] = crop_data[feature_cols].corr()
            
            self.crop_profiles[crop] = profile
            print(f"✓ Learned profile for {crop} from {n_samples} samples")
            print(f"  Example std devs: pH={profile['stds']['soil_ph']:.2f}, humidity={profile['stds']['humidity']:.2f}")
    
    def generate_from_learned_profile(self, crop_name, n_samples):
        """Generate synthetic data based on learned crop profile"""
        if crop_name not in self.crop_profiles:
            raise ValueError(f"No learned profile for {crop_name}. Available: {list(self.crop_profiles.keys())}")
        
        profile = self.crop_profiles[crop_name]
        feature_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
        
        if profile['correlations'] is not None and self.realism_mode:
            means = [profile['means'][col] for col in feature_cols]
            stds = [profile['stds'][col] for col in feature_cols]
            cov_matrix = np.outer(stds, stds) * profile['correlations'].values
            
            samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
            
            for i, col in enumerate(feature_cols):
                min_val, max_val = profile['ranges'][col]
                samples[:, i] = np.clip(samples[:, i], min_val, max_val)
        else:
            samples = np.zeros((n_samples, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                mean = profile['means'][col]
                std = profile['stds'][col]
                min_val, max_val = profile['ranges'][col]
                
                samples[:, i] = np.random.normal(mean, std, n_samples)
                samples[:, i] = np.clip(samples[:, i], min_val, max_val)
        
        # FIXED: Use appropriate decimal places to preserve variation
        samples[:, 0] = np.round(samples[:, 0], 2)  # soil_ph - 2 decimals
        samples[:, 1] = np.round(samples[:, 1], 1)  # fertility_ec - 1 decimal (was 0!)
        samples[:, 2] = np.round(samples[:, 2], 1)  # humidity - 1 decimal (was 0!)
        samples[:, 3] = np.round(samples[:, 3], 1)  # sunlight - 1 decimal (was 0!)
        samples[:, 4] = np.round(samples[:, 4], 2)  # soil_temp - 2 decimals
        samples[:, 5] = np.round(samples[:, 5], 1)  # soil_moisture - 1 decimal (was 0!)
        
        df = pd.DataFrame(samples, columns=feature_cols)
        df['label'] = crop_name
        
        return df
    
    def add_synthetic_samples(self, crop_name, n_samples):
        """Add synthetic samples for a crop based on learned profile"""
        synthetic_data = self.generate_from_learned_profile(crop_name, n_samples)
        self.existing_data = pd.concat([self.existing_data, synthetic_data], ignore_index=True)
        print(f"✓ Added {n_samples} synthetic samples for {crop_name}")
    
    def generate_manual_crop_data(self, crop_name, n_samples,
                                  ph_range, ec_range, humidity_range,
                                  sunlight_range, soil_temp_range, soil_moisture_range,
                                  ph_mean=None, ec_mean=None, humidity_mean=None,
                                  sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """Generate synthetic data using manual parameters"""
        if ph_mean is None: ph_mean = (ph_range[0] + ph_range[1]) / 2
        if ec_mean is None: ec_mean = (ec_range[0] + ec_range[1]) / 2
        if humidity_mean is None: humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if sunlight_mean is None: sunlight_mean = (sunlight_range[0] + sunlight_range[1]) / 2
        if soil_temp_mean is None: soil_temp_mean = (soil_temp_range[0] + soil_temp_range[1]) / 2
        if soil_moisture_mean is None: soil_moisture_mean = (soil_moisture_range[0] + soil_moisture_range[1]) / 2
        
        rows = []
        for _ in range(n_samples):
            # FIXED: Use 20% of range for std instead of 5-15%
            soil_ph = round(np.clip(np.random.normal(ph_mean, (ph_range[1] - ph_range[0]) * 0.20), *ph_range), 2)
            fertility_ec = round(np.clip(np.random.normal(ec_mean, (ec_range[1] - ec_range[0]) * 0.20), *ec_range), 1)
            humidity = round(np.clip(np.random.normal(humidity_mean, (humidity_range[1] - humidity_range[0]) * 0.15), *humidity_range), 1)
            soil_temp = round(np.clip(np.random.normal(soil_temp_mean, (soil_temp_range[1] - soil_temp_range[0]) * 0.15), *soil_temp_range), 2)
            
            # Correlations with more variation
            if self.realism_mode:
                sunlight_variation = (sunlight_range[1] - sunlight_range[0]) * 0.25
                sunlight_adj_mean = sunlight_mean * (0.85 + 0.30 * (soil_temp - soil_temp_range[0]) / (soil_temp_range[1] - soil_temp_range[0]))
                sunlight = round(np.clip(np.random.normal(sunlight_adj_mean, sunlight_variation), *sunlight_range), 1)
                
                moisture_variation = (soil_moisture_range[1] - soil_moisture_range[0]) * 0.20
                soil_moisture_adj_mean = soil_moisture_mean * (0.85 + 0.30 * (humidity - humidity_range[0]) / (humidity_range[1] - humidity_range[0]))
                soil_moisture = round(np.clip(np.random.normal(soil_moisture_adj_mean, moisture_variation), *soil_moisture_range), 1)
            else:
                sunlight = round(np.clip(np.random.normal(sunlight_mean, (sunlight_range[1] - sunlight_range[0]) * 0.25), *sunlight_range), 1)
                soil_moisture = round(np.clip(np.random.normal(soil_moisture_mean, (soil_moisture_range[1] - soil_moisture_range[0]) * 0.20), *soil_moisture_range), 1)
            
            rows.append([soil_ph, fertility_ec, humidity, sunlight, soil_temp, soil_moisture, crop_name])
        
        df = pd.DataFrame(rows, columns=['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label'])
        return df
    
    def fill_crop_to_target(self, crop_name, target_count,
                           ph_range=None, ec_range=None, humidity_range=None,
                           sunlight_range=None, soil_temp_range=None, soil_moisture_range=None,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """
        Ensure a crop reaches target sample size.
        
        - If crop has learned profile: uses learned data (ignores manual parameters)
        - If crop has no profile: requires manual parameters
        """
        current_count = self.get_crop_count(crop_name)
        
        if current_count >= target_count:
            print(f"✓ {crop_name} already has {current_count} samples (target: {target_count})")
            return
        
        needed = target_count - current_count
        
        if crop_name in self.crop_profiles:
            print(f"→ Generating {needed} samples for {crop_name} using learned profile ({current_count} → {target_count})")
            self.add_synthetic_samples(crop_name, needed)
        else:
            required_params = [ph_range, ec_range, humidity_range, sunlight_range, soil_temp_range, soil_moisture_range]
            if any(p is None for p in required_params):
                print(f"✗ Cannot generate for {crop_name}: no learned profile and missing manual parameters")
                return
            
            print(f"→ Generating {needed} samples for {crop_name} using manual parameters ({current_count} → {target_count})")
            manual_data = self.generate_manual_crop_data(
                crop_name, needed,
                ph_range, ec_range, humidity_range,
                sunlight_range, soil_temp_range, soil_moisture_range,
                ph_mean, ec_mean, humidity_mean,
                sunlight_mean, soil_temp_mean, soil_moisture_mean
            )
            self.existing_data = pd.concat([self.existing_data, manual_data], ignore_index=True)
            print(f"✓ Added {needed} manual samples for {crop_name}")
    
    def get_single_sample_crops(self):
        """Return list of crops with only 1 sample"""
        single_sample_crops = []
        for crop in self.existing_data['label'].unique():
            if self.get_crop_count(crop) == 1:
                single_sample_crops.append(crop)
        return single_sample_crops
    
    def export_single_samples_as_template(self, output_path="single_sample_template.py"):
        """
        Export crops with 1 sample as a Python template for manual parameter entry.
        This helps you quickly create fill_crop_to_target() calls for all single-sample crops.
        """
        single_crops = self.get_single_sample_crops()
        
        if not single_crops:
            print("✓ No single-sample crops found")
            return
        
        feature_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
        
        template_lines = [
            "# Template for crops with only 1 sample",
            "# Copy these into your main script and adjust ranges as needed",
            "# The values shown are from your single sample\n"
        ]
        
        for crop in single_crops:
            crop_data = self.existing_data[self.existing_data['label'] == crop].iloc[0]
            
            template_lines.append(f"# {crop}")
            template_lines.append(f"generator.fill_crop_to_target(")
            template_lines.append(f'    crop_name="{crop}",')
            template_lines.append(f'    target_count=100,  # Adjust as needed')
            
            # Use single sample values as means, suggest reasonable ranges
            for col in feature_cols:
                value = crop_data[col]
                
                # Suggest ranges based on typical variations
                if col == 'soil_ph':
                    range_size = 1.0
                    mean_val = round(value, 1)
                elif col == 'fertility_ec':
                    range_size = value * 0.3
                    mean_val = round(value)
                elif col in ['humidity', 'soil_moisture']:
                    range_size = 15
                    mean_val = round(value)
                elif col == 'sunlight':
                    range_size = value * 0.4
                    mean_val = round(value)
                elif col == 'soil_temp':
                    range_size = 3.0
                    mean_val = round(value, 1)
                else:
                    range_size = value * 0.2
                    mean_val = round(value)
                
                min_val = max(0, mean_val - range_size / 2)
                max_val = mean_val + range_size / 2
                
                col_formatted = col.replace('soil_', '').replace('fertility_', '')
                template_lines.append(f'    {col_formatted}_range=({min_val:.1f}, {max_val:.1f}), {col_formatted}_mean={mean_val},')
            
            template_lines.append(")\n")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(template_lines))
        
        print(f"✓ Template exported to {output_path}")
        print(f"  Found {len(single_crops)} crops with 1 sample: {', '.join(single_crops)}")
        print(f"  Review and adjust the ranges, then copy into your main script")
    
    def get_crop_count(self, crop_name):
        """Count samples for crop"""
        if len(self.existing_data) == 0:
            return 0
        return len(self.existing_data[self.existing_data['label'] == crop_name])
    
    def save_dataset(self, output_path, include_original=True):
        """
        Save dataset to CSV
        
        Parameters:
        output_path (str): Output file path
        include_original (bool): If False, only saves synthetic data (excludes original rows)
        """
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        if include_original:
            self.existing_data.to_csv(output_path, index=False)
            print(f"✓ Dataset saved to {output_path} (original + synthetic)")
        else:
            # Load original to identify synthetic rows
            if self.existing_csv_path and os.path.exists(self.existing_csv_path):
                original = pd.read_csv(self.existing_csv_path)
                original_count = len(original)
                synthetic_only = self.existing_data.iloc[original_count:]
                synthetic_only.to_csv(output_path, index=False)
                print(f"✓ Synthetic data saved to {output_path} ({len(synthetic_only)} rows)")
            else:
                self.existing_data.to_csv(output_path, index=False)
                print(f"✓ Dataset saved to {output_path}")
    
    def show_dataset_summary(self):
        """Show summary of dataset"""
        if len(self.existing_data) == 0:
            print("Dataset is empty")
            return
        
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total rows: {len(self.existing_data)}")
        print(f"Unique crops: {len(self.existing_data['label'].unique())}")
        print("\nCrop distribution:")
        print(self.existing_data['label'].value_counts().to_string())
        print("="*50 + "\n")
    
    def show_crop_profile(self, crop_name):
        """Display learned profile for a crop"""
        if crop_name not in self.crop_profiles:
            print(f"No profile for {crop_name}")
            return
        
        profile = self.crop_profiles[crop_name]
        print(f"\nProfile for {crop_name}:")
        print("-" * 60)
        print(f"{'Feature':<15} {'Mean':>10} {'Std Dev':>10} {'Range':>20}")
        print("-" * 60)
        for feature in profile['means'].keys():
            mean = profile['means'][feature]
            std = profile['stds'][feature]
            rng = profile['ranges'][feature]
            print(f"{feature:<15} {mean:>10.2f} {std:>10.2f} {rng[0]:>9.2f}-{rng[1]:<9.2f}")
        print("-" * 60)


# Example usage
if __name__ == "__main__":
    generator = SmartCropDataGenerator(
        existing_csv_path="../dataset/SPC-soil-data.csv",
        realism_mode=True
    )
    
    if generator.crop_profiles:
        print("\n" + "="*50)
        print("LEARNED PROFILES:")
        print("="*50)
        for crop in list(generator.crop_profiles.keys())[:3]:  # Show first 3
            generator.show_crop_profile(crop)
    
    # Export template for single-sample crops
    print("\n" + "="*50)
    print("CHECKING FOR SINGLE-SAMPLE CROPS:")
    print("="*50)
    generator.export_single_samples_as_template("../dataset/single_sample_template.py")
    
    # OPTION 1: Generate for existing crops with 2+ samples (uses learned profiles)
    print("\n" + "="*50)
    print("AUGMENTING CROPS WITH LEARNED PROFILES:")
    print("="*50)
    target_samples = 100  # Start with 100 samples per crop
    for crop in generator.crop_profiles.keys():
        generator.fill_crop_to_target(crop, target_samples)
    
    # OPTION 2: Add parameters for single-sample crops
    # After running once, check single_sample_template.py
    # Then copy the generated code here and adjust ranges as needed
    
    # Example for crops you add manually:
    # Cacao - understory crop, shade-tolerant, very sensitive to drought
   
    

    # # Guyabano - sun-tolerant, prefers well-drained soils
    # generator.fill_crop_to_target(
    #     crop_name="Guyabano",
    #     target_count=100,
    #     ph_range=(5.5, 7.0), ph_mean=6.3,
    #     ec_range=(1000, 1800), ec_mean=1400,
    #     humidity_range=(70, 90), humidity_mean=80,
    #     sunlight_range=(40000, 70000), sunlight_mean=55000,  # high sun
    #     soil_temp_range=(24, 30), soil_temp_mean=27,
    #     soil_moisture_range=(55, 75), soil_moisture_mean=65
    # )

 

    # # Durian - deep-rooted, high water demand, sun-loving
    # generator.fill_crop_to_target(
    #     crop_name="Durian",
    #     target_count=100,
    #     ph_range=(5.0, 6.5), ph_mean=5.7,
    #     ec_range=(1200, 2000), ec_mean=1600,
    #     humidity_range=(75, 95), humidity_mean=85,
    #     sunlight_range=(40000, 70000), sunlight_mean=55000,  # high sun
    #     soil_temp_range=(24, 30), soil_temp_mean=27,
    #     soil_moisture_range=(65, 85), soil_moisture_mean=75
    # )
 
    # generator.fill_crop_to_target(
    #     crop_name="Orange",
    #     target_count=100,
    #     ph_range=(5.5, 7.2), ph_mean=6.5,
    #     ec_range=(1000, 1800), ec_mean=1400,
    #     humidity_range=(50, 75), humidity_mean=65,   # lower humidity tolerance
    #     sunlight_range=(40000, 85000), sunlight_mean=65000,
    #     soil_temp_range=(18, 28), soil_temp_mean=23,   
    #     soil_moisture_range=(45, 65), soil_moisture_mean=55
    # ) 
    # generator.fill_crop_to_target(
    #     crop_name="Snap Bean",
    #     target_count=100,
    #     ph_range=(6.0, 7.0), ph_mean=6.5,
    #     ec_range=(500, 800), ec_mean=600,
    #     humidity_range=(69, 81), humidity_mean=77,
    #     sunlight_range=(1100, 13100), sunlight_mean=7100,  
    #     soil_temp_range=(23, 29), soil_temp_mean=26,         
    #     soil_moisture_range=(84, 94), soil_moisture_mean=89
    # )

   

    # # Sigarilyas (Winged Bean) - tropical, rainfall-adapted
    # generator.fill_crop_to_target(
    #     crop_name="Sigarilyas",
    #     target_count=100,
    #     ph_range=(5.5, 6.8), ph_mean=6.2,
    #     ec_range=(1200, 2000), ec_mean=1600,
    #     humidity_range=(65, 85), humidity_mean=75,
    #     sunlight_range=(40000, 80000), sunlight_mean=60000,
    #     soil_temp_range=(22, 30), soil_temp_mean=26,
    #     soil_moisture_range=(55, 80), soil_moisture_mean=68  # higher moisture demand
    # )

    # # Mungbean - drought-tolerant, low fertility needs
    # generator.fill_crop_to_target(
    #     crop_name="Mungbean",
    #     target_count=100,
    #     ph_range=(5.5, 7.0), ph_mean=6.2,
    #     ec_range=(800, 1600), ec_mean=1200,                  # less fertilizer needed
    #     humidity_range=(50, 75), humidity_mean=65,           # can grow in drier air
    #     sunlight_range=(45000, 90000), sunlight_mean=70000,
    #     soil_temp_range=(22, 32), soil_temp_mean=27,
    #     soil_moisture_range=(35, 65), soil_moisture_mean=50  # lowest water demand in group
    # ) 
    # generator.fill_crop_to_target(
    #     crop_name="Mustard",
    #     target_count=100,
    #     ph_range=(5.5, 7.0), ph_mean=6.3,
    #     ec_range=(800, 1600), ec_mean=1200,                 # lower fertility need
    #     humidity_range=(50, 75), humidity_mean=62,          # tolerates drier air
    #     sunlight_range=(30000, 80000), sunlight_mean=55000, # full sun tolerant
    #     soil_temp_range=(15, 25), soil_temp_mean=20,        # prefers cooler soils
    #     soil_moisture_range=(40, 70), soil_moisture_mean=55 # tolerates drier soil
    # ) 
    # generator.fill_crop_to_target(
    #     crop_name="Patola",
    #     target_count=100,
    #     ph_range=(5.5, 6.5), ph_mean=6.0,
    #     ec_range=(1500, 2500), ec_mean=2000,                 # higher fertility demand
    #     humidity_range=(60, 85), humidity_mean=75,           # prefers more humidity
    #     sunlight_range=(50000, 90000), sunlight_mean=70000,  # loves strong sun
    #     soil_temp_range=(22, 30), soil_temp_mean=26,
    #     soil_moisture_range=(55, 80), soil_moisture_mean=68  # consistent water needed
    # ) 
    # generator.fill_crop_to_target(
    #     crop_name="Sili Panigang",
    #     target_count=100,
    #     ph_range=(5.4, 6.6), ph_mean=6.0,
    #     ec_range=(430, 570), ec_mean=500,                
    #     humidity_range=(50, 75), humidity_mean=62,            
    #     sunlight_range=(2000, 9000), sunlight_mean=5000,   
    #     soil_temp_range=(27.1, 31.5), soil_temp_mean=29,       
    #     soil_moisture_range=(81.2, 88.4), soil_moisture_mean=84 
    # )

    # # Sili Tingala (Bird’s Eye Chili) - hot pepper, stress-tolerant
    # generator.fill_crop_to_target(
    #     crop_name="Sili Tingala",
    #     target_count=100,
    #     ph_range=(5.5, 6.5), ph_mean=6.0,
    #     ec_range=(440, 560), ec_mean=500,                
    #     humidity_range=(50, 75), humidity_mean=62,            
    #     sunlight_range=(2000, 9000), sunlight_mean=5000,   
    #     soil_temp_range=(27.1, 31.5), soil_temp_mean=29,       
    #     soil_moisture_range=(81, 88), soil_moisture_mean=84 
    # ) 

    # # Watermelon - drought-tolerant cucurbit, wide pH range, less fertility demand
    # generator.fill_crop_to_target(
    #     crop_name="Watermelon",
    #     target_count=100,
    #     ph_range=(5.5, 7.5), ph_mean=6.5,                 # widest tolerance here
    #     ec_range=(800, 1600), ec_mean=1200,               # lower fertility demand
    #     humidity_range=(50, 70), humidity_mean=60,        # prefers drier air
    #     sunlight_range=(50000, 95000), sunlight_mean=75000,
    #     soil_temp_range=(22, 32), soil_temp_mean=27,
    #     soil_moisture_range=(40, 70), soil_moisture_mean=55  # more drought tolerant
    # ) 
    
    #     # Tomato - prefers cooler root zone, high fertility, consistent moisture
    # generator.fill_crop_to_target(
    #     crop_name="Tomato",
    #     target_count=100,
    #     ph_range=(5.5, 6.8), ph_mean=6.2,
    #     ec_range=(2000, 3000), ec_mean=2400,            # high nutrient demand
    #     humidity_range=(55, 75), humidity_mean=68,      # moderate humidity
    #     sunlight_range=(45000, 85000), sunlight_mean=65000,
    #     soil_temp_range=(16, 26), soil_temp_mean=22,    # cooler root temp preferred
    #     soil_moisture_range=(60, 80), soil_moisture_mean=70  # steady moisture for fruit set
    # ) 

    # # print("Cool weather: onion, radish")


    #     # Onion - prefers cool to mild temps, moderate fertility, sensitive to waterlogging
    # generator.fill_crop_to_target(
    #     crop_name="Onion",
    #     target_count=100,
    #     ph_range=(6.0, 7.0), ph_mean=6.5,
    #     ec_range=(1500, 2500), ec_mean=2000,
    #     humidity_range=(50, 70), humidity_mean=60,
    #     sunlight_range=(40000, 80000), sunlight_mean=60000,
    #     soil_temp_range=(15, 25), soil_temp_mean=20,      # cool-moderate
    #     soil_moisture_range=(40, 65), soil_moisture_mean=52  # dislikes excess water
    # )

    # # Radish - faster growing, thrives in cooler soil, needs steadier moisture
    # generator.fill_crop_to_target(
    #     crop_name="Radish",
    #     target_count=100,
    #     ph_range=(6.0, 7.0), ph_mean=6.5,
    #     ec_range=(1200, 2200), ec_mean=1700,              # slightly lower fertility need
    #     humidity_range=(50, 75), humidity_mean=62,        # tolerates a touch more humidity
    #     sunlight_range=(35000, 75000), sunlight_mean=55000,
    #     soil_temp_range=(10, 20), soil_temp_mean=15,      # cooler than onion
    #     soil_moisture_range=(50, 75), soil_moisture_mean=65  # steady moisture for root swelling
    # ) 

    # # Apple - temperate, requires chilling hours, cool soil temps
    # generator.fill_crop_to_target(
    #     crop_name="Apple",
    #     target_count=100,
    #     ph_range=(5.5, 7.0), ph_mean=6.3,
    #     ec_range=(800, 1600), ec_mean=1200,
    #     humidity_range=(45, 65), humidity_mean=55,        # drier temperate air
    #     sunlight_range=(30000, 70000), sunlight_mean=50000,
    #     soil_temp_range=(10, 20), soil_temp_mean=15,      # cooler soil temps
    #     soil_moisture_range=(45, 65), soil_moisture_mean=55
    # )

    # # Grapes - Mediterranean climate crop, prefers warm temps, lots of sun, tolerates drought
    # generator.fill_crop_to_target(
    #     crop_name="Grapes",
    #     target_count=100,
    #     ph_range=(5.5, 6.8), ph_mean=6.2,
    #     ec_range=(800, 1600), ec_mean=1200,
    #     humidity_range=(40, 60), humidity_mean=50,        # needs drier climate to avoid rot
    #     sunlight_range=(55000, 100000), sunlight_mean=80000,
    #     soil_temp_range=(18, 28), soil_temp_mean=23,
    #     soil_moisture_range=(40, 60), soil_moisture_mean=50  # drought tolerant
    # )

   

    # # Oyster Mushroom - shade crop, very high humidity, moderate temps
    # generator.fill_crop_to_target(
    #     crop_name="Oyster Mushroom",
    #     target_count=100,
    #     ph_range=(5.5, 6.5), ph_mean=6.0,
    #     ec_range=(500, 1500), ec_mean=1000,
    #     humidity_range=(85, 95), humidity_mean=90,        # nearly saturated air
    #     sunlight_range=(2000, 10000), sunlight_mean=6000, # low light / shade
    #     soil_temp_range=(18, 25), soil_temp_mean=22,
    #     soil_moisture_range=(65, 90), soil_moisture_mean=78
    # ) 
   
 

    # # Okra - heat-loving, full-sun annual; tolerates drier spells but performs with moderate fertility
    # generator.fill_crop_to_target(
    #     crop_name="Okra",
    #     target_count=100,
    #     ph_range=(5.5, 7.0), ph_mean=6.2,
    #     ec_range=(1200, 2200), ec_mean=1700,           # moderate-to-high fertility
    #     humidity_range=(60, 85), humidity_mean=72,
    #     sunlight_range=(60000, 100000), sunlight_mean=80000,
    #     soil_temp_range=(22, 34), soil_temp_mean=28,
    #     soil_moisture_range=(40, 75), soil_moisture_mean=55  # tolerates some dryness
    # )



    # single sample 

        # Template for crops with only 1 sample
    # Copy these into your main script and adjust ranges as needed
    # The values shown are from your single sample

    # Bamboo
    generator.fill_crop_to_target(
        crop_name="Bamboo",
        target_count=100,  # Adjust as needed
        ph_range=(3.5, 4.5), ph_mean=4.0,
        ec_range=(537.2, 726.8), ec_mean=632,
        humidity_range=(54.5, 69.5), humidity_mean=62,
        sunlight_range=(6461.6, 9692.4), sunlight_mean=8077,
       soil_temp_range=(28.0, 31.0),soil_temp_mean=29.5,
       soil_moisture_range=(91.5, 106.5),soil_moisture_mean=99,
    )

    # Black Pepper
    generator.fill_crop_to_target(
        crop_name="Black Pepper",
        target_count=100,  # Adjust as needed
        ph_range=(5.1, 6.1), ph_mean=5.6,
        ec_range=(374.0, 506.0), ec_mean=440,
        humidity_range=(77.5, 92.5), humidity_mean=85,
        sunlight_range=(720.0, 1080.0), sunlight_mean=900,
       soil_temp_range=(25.4, 28.4),soil_temp_mean=26.9,
       soil_moisture_range=(77.5, 92.5),soil_moisture_mean=85,
    )

   
    # Cassava
    generator.fill_crop_to_target(
        crop_name="Cassava",
        target_count=100,  # Adjust as needed
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(439.4, 594.5), ec_mean=517,
        humidity_range=(80.5, 95.5), humidity_mean=88,
        sunlight_range=(875.2, 1312.8), sunlight_mean=1094,
       soil_temp_range=(27.0, 30.0),soil_temp_mean=28.5,
       soil_moisture_range=(91.5, 106.5),soil_moisture_mean=99,
    )

    # Cowpea
    generator.fill_crop_to_target(
        crop_name="Cowpea",
        target_count=100,  # Adjust as needed
        ph_range=(5.7, 6.7), ph_mean=6.2,
        ec_range=(348.5, 471.5), ec_mean=410,
        humidity_range=(70.5, 85.5), humidity_mean=78,
        sunlight_range=(1760.0, 2640.0), sunlight_mean=2200,
       soil_temp_range=(26.9, 29.9),soil_temp_mean=28.4,
       soil_moisture_range=(75.5, 90.5),soil_moisture_mean=83,
    )

    # Forage Grass
    generator.fill_crop_to_target(
        crop_name="Forage Grass",
        target_count=100,  # Adjust as needed
        ph_range=(6.3, 7.3), ph_mean=6.8,
        ec_range=(344.2, 465.8), ec_mean=405,
        humidity_range=(71.5, 86.5), humidity_mean=79,
        sunlight_range=(1520.0, 2280.0), sunlight_mean=1900,
       soil_temp_range=(26.5, 29.5),soil_temp_mean=28.0,
       soil_moisture_range=(76.5, 91.5),soil_moisture_mean=84,
    )

    # Ipil Ipil
    generator.fill_crop_to_target(
        crop_name="Ipil Ipil",
        target_count=100,  # Adjust as needed
        ph_range=(6.3, 7.3), ph_mean=6.8,
        ec_range=(344.2, 465.8), ec_mean=405,
        humidity_range=(71.5, 86.5), humidity_mean=79,
        sunlight_range=(1520.0, 2280.0), sunlight_mean=1900,
       soil_temp_range=(26.5, 29.5),soil_temp_mean=28.0,
       soil_moisture_range=(76.5, 91.5),soil_moisture_mean=84,
    )

    # Jackfruit
    generator.fill_crop_to_target(
        crop_name="Jackfruit",
        target_count=100,  # Adjust as needed
        ph_range=(5.2, 6.2), ph_mean=5.7,
        ec_range=(399.5, 540.5), ec_mean=470,
        humidity_range=(76.5, 91.5), humidity_mean=84,
        sunlight_range=(840.0, 1260.0), sunlight_mean=1050,
       soil_temp_range=(25.9, 28.9),soil_temp_mean=27.4,
       soil_moisture_range=(81.5, 96.5),soil_moisture_mean=89,
    )

    # Rice
    generator.fill_crop_to_target(
        crop_name="Rice",
        target_count=100,  # Adjust as needed
        ph_range=(6.3, 7.3), ph_mean=6.8,
        ec_range=(467.5, 632.5), ec_mean=550,
        humidity_range=(67.5, 82.5), humidity_mean=75,
        sunlight_range=(2000.0, 3000.0), sunlight_mean=2500,
       soil_temp_range=(27.7, 30.7),soil_temp_mean=29.2,
       soil_moisture_range=(87.5, 102.5),soil_moisture_mean=95,
    )

    # Kamoteng Baging
    generator.fill_crop_to_target(
        crop_name="Kamoteng Baging",
        target_count=100,  # Adjust as needed
        ph_range=(4.0, 5.0), ph_mean=4.5,
        ec_range=(305.1, 412.9), ec_mean=359,
        humidity_range=(62.5, 77.5), humidity_mean=70,
        sunlight_range=(12671.2, 19006.8), sunlight_mean=15839,
       soil_temp_range=(33.1, 36.1),soil_temp_mean=34.6,
       soil_moisture_range=(91.5, 106.5),soil_moisture_mean=99,
    )

    # Katuray
    generator.fill_crop_to_target(
        crop_name="Katuray",
        target_count=100,  # Adjust as needed
        ph_range=(5.7, 6.7), ph_mean=6.2,
        ec_range=(348.5, 471.5), ec_mean=410,
        humidity_range=(70.5, 85.5), humidity_mean=78,
        sunlight_range=(1760.0, 2640.0), sunlight_mean=2200,
       soil_temp_range=(26.9, 29.9),soil_temp_mean=28.4,
       soil_moisture_range=(75.5, 90.5),soil_moisture_mean=83,
    )

    # Kulo
    generator.fill_crop_to_target(
        crop_name="Kulo",
        target_count=100,  # Adjust as needed
        ph_range=(5.2, 6.2), ph_mean=5.7,
        ec_range=(399.5, 540.5), ec_mean=470,
        humidity_range=(76.5, 91.5), humidity_mean=84,
        sunlight_range=(840.0, 1260.0), sunlight_mean=1050,
       soil_temp_range=(25.9, 28.9),soil_temp_mean=27.4,
       soil_moisture_range=(81.5, 96.5),soil_moisture_mean=89,
    )

    # Lipote
    generator.fill_crop_to_target(
        crop_name="Lipote",
        target_count=100,  # Adjust as needed
        ph_range=(5.2, 6.2), ph_mean=5.7,
        ec_range=(357.0, 483.0), ec_mean=420,
        humidity_range=(73.5, 88.5), humidity_mean=81,
        sunlight_range=(760.0, 1140.0), sunlight_mean=950,
       soil_temp_range=(26.5, 29.5),soil_temp_mean=28.0,
       soil_moisture_range=(72.5, 87.5),soil_moisture_mean=80,
    )

    # Orchid
    generator.fill_crop_to_target(
        crop_name="Orchid",
        target_count=100,  # Adjust as needed
        ph_range=(5.6, 6.6), ph_mean=6.1,
        ec_range=(331.5, 448.5), ec_mean=390,
        humidity_range=(68.5, 83.5), humidity_mean=76,
        sunlight_range=(1480.0, 2220.0), sunlight_mean=1850,
       soil_temp_range=(26.8, 29.8),soil_temp_mean=28.3,
       soil_moisture_range=(70.5, 85.5),soil_moisture_mean=78,
    )

    # Papaya
    generator.fill_crop_to_target(
        crop_name="Papaya",
        target_count=100,  # Adjust as needed
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(369.8, 500.2), ec_mean=435,
        humidity_range=(71.5, 86.5), humidity_mean=79,
        sunlight_range=(1600.0, 2400.0), sunlight_mean=2000,
       soil_temp_range=(26.6, 29.6),soil_temp_mean=28.1,
       soil_moisture_range=(74.5, 89.5),soil_moisture_mean=82,
    )

    # Pechay
    generator.fill_crop_to_target(
        crop_name="Pechay",
        target_count=100,  # Adjust as needed
        ph_range=(6.1, 7.1), ph_mean=6.6,
        ec_range=(454.8, 615.2), ec_mean=535,
        humidity_range=(67.5, 82.5), humidity_mean=75,
        sunlight_range=(2080.0, 3120.0), sunlight_mean=2600,
       soil_temp_range=(27.2, 30.2),soil_temp_mean=28.7,
       soil_moisture_range=(86.5, 101.5),soil_moisture_mean=94,
    )

    # Sili Labuyo
    generator.fill_crop_to_target(
        crop_name="Sili Labuyo",
        target_count=100,  # Adjust as needed
        ph_range=(5.0, 6.0), ph_mean=5.5,
        ec_range=(412.2, 557.8), ec_mean=485,
        humidity_range=(68.5, 83.5), humidity_mean=76,
        sunlight_range=(1920.0, 2880.0), sunlight_mean=2400,
       soil_temp_range=(27.4, 30.4),soil_temp_mean=28.9,
       soil_moisture_range=(79.5, 94.5),soil_moisture_mean=87,
    )

    # Sweet Potato
    generator.fill_crop_to_target(
        crop_name="Sweet Potato",
        target_count=100,  # Adjust as needed
        ph_range=(5.9, 6.9), ph_mean=6.4,
        ec_range=(420.8, 569.2), ec_mean=495,
        humidity_range=(73.5, 88.5), humidity_mean=81,
        sunlight_range=(1320.0, 1980.0), sunlight_mean=1650,
       soil_temp_range=(26.3, 29.3),soil_temp_mean=27.8,
       soil_moisture_range=(84.5, 99.5),soil_moisture_mean=92,
    )

    # Sweet Sorghum
    generator.fill_crop_to_target(
        crop_name="Sweet Sorghum",
        target_count=100,  # Adjust as needed
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(437.8, 592.2), ec_mean=515,
        humidity_range=(70.5, 85.5), humidity_mean=78,
        sunlight_range=(2200.0, 3300.0), sunlight_mean=2750,
       soil_temp_range=(26.9, 29.9),soil_temp_mean=28.4,
       soil_moisture_range=(81.5, 96.5),soil_moisture_mean=89,
    )

    # String Bean
    generator.fill_crop_to_target(
        crop_name="String Bean",
        target_count=100,  # Adjust as needed
        ph_range=(6.4, 7.4), ph_mean=6.9,
        ec_range=(459.0, 621.0), ec_mean=540,
        humidity_range=(68.5, 83.5), humidity_mean=76,
        sunlight_range=(2160.0, 3240.0), sunlight_mean=2700,
       soil_temp_range=(27.5, 30.5),soil_temp_mean=29.0,
       soil_moisture_range=(85.5, 100.5),soil_moisture_mean=93,
    )

    # Upo
    generator.fill_crop_to_target(
        crop_name="Upo",
        target_count=100,  # Adjust as needed
        ph_range=(6.4, 7.4), ph_mean=6.9,
        ec_range=(459.0, 621.0), ec_mean=540,
        humidity_range=(68.5, 83.5), humidity_mean=76,
        sunlight_range=(2160.0, 3240.0), sunlight_mean=2700,
       soil_temp_range=(27.5, 30.5),soil_temp_mean=29.0,
       soil_moisture_range=(85.5, 100.5),soil_moisture_mean=93,
    )

    # Ube
    generator.fill_crop_to_target(
        crop_name="Ube",
        target_count=100,  # Adjust as needed
        ph_range=(4.9, 5.9), ph_mean=5.4,
        ec_range=(399.5, 540.5), ec_mean=470,
        humidity_range=(75.5, 90.5), humidity_mean=83,
        sunlight_range=(1040.0, 1560.0), sunlight_mean=1300,
       soil_temp_range=(26.4, 29.4),soil_temp_mean=27.9,
       soil_moisture_range=(82.5, 97.5),soil_moisture_mean=90,
    )




    
    # Show summary
    generator.show_dataset_summary()
    
    # Save with original data included (back to dataset folder)
    generator.save_dataset("../dataset/augmented_crop_data.csv", include_original=True)
    
    # Or save only synthetic data
    # generator.save_dataset("../dataset/synthetic_only.csv", include_original=False)