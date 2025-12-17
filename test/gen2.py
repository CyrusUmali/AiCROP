import pandas as pd
import numpy as np
import os
import sys

class CropDataGenerator:
    def __init__(self, existing_csv_path=None, spc_data_path=None, realism_mode=True):
        """
        Initialize the crop data generator
        
        Parameters:
        existing_csv_path (str): Path to the existing enhanced CSV file (optional)
        spc_data_path (str): Path to the original SPC data CSV file
        realism_mode (bool): If True, adjusts generated data to be closer to realistic agricultural ranges
        """
        self.existing_csv_path = existing_csv_path
        self.spc_data_path = spc_data_path
        self.realism_mode = realism_mode
        
        # Keep SPC and enhanced data separate
        self.spc_data = None
        self.enhanced_data = pd.DataFrame(
            columns=['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
        )
        
        # Load SPC data first (if available)
        if spc_data_path:
            self.load_spc_data()
        
        # Load enhanced data if exists
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
    
    def load_spc_data(self):
        """Load original SPC soil data from parent folder"""
        try:
            # Go up one level from test folder to dataset folder
            if os.path.basename(os.getcwd()) == 'test':
                # We're in test folder, go up to parent
                parent_dir = os.path.dirname(os.getcwd())
                spc_full_path = os.path.join(parent_dir, 'dataset', self.spc_data_path)
            else:
                spc_full_path = os.path.join('dataset', self.spc_data_path)
            
            if os.path.exists(spc_full_path):
                spc_data = pd.read_csv(spc_full_path)
                print(f"SPC data columns: {list(spc_data.columns)}")
                
                # Check if columns match our expected format
                # Assuming SPC data might have different column names
                # Let's map common column names
                column_mapping = {}
                
                # Try to automatically map columns
                for col in spc_data.columns:
                    col_lower = col.lower()
                    if 'ph' in col_lower:
                        column_mapping[col] = 'soil_ph'
                    elif 'ec' in col_lower or 'fertility' in col_lower:
                        column_mapping[col] = 'fertility_ec'
                    elif 'humid' in col_lower:
                        column_mapping[col] = 'humidity'
                    elif 'sun' in col_lower or 'light' in col_lower:
                        column_mapping[col] = 'sunlight'
                    elif 'temp' in col_lower and 'soil' in col_lower:
                        column_mapping[col] = 'soil_temp'
                    elif 'moist' in col_lower and 'soil' in col_lower:
                        column_mapping[col] = 'soil_moisture'
                    elif 'crop' in col_lower or 'label' in col_lower:
                        column_mapping[col] = 'label'
                
                # If we found mappings, rename columns
                if column_mapping:
                    spc_data = spc_data.rename(columns=column_mapping)
                    # Select only the columns we need
                    expected_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 
                                   'soil_temp', 'soil_moisture', 'label']
                    available_cols = [col for col in expected_cols if col in spc_data.columns]
                    
                    if 'label' in available_cols and len(available_cols) >= 4:
                        spc_data = spc_data[available_cols]
                        self.spc_data = spc_data
                        print(f"Loaded SPC data: {len(spc_data)} rows")
                        print(f"SPC crops: {list(spc_data['label'].unique())}")
                    else:
                        print("Warning: SPC data doesn't have required columns after mapping")
                else:
                    print("Warning: Could not map SPC data columns to expected format")
                    
            else:
                print(f"Warning: SPC data file not found at {spc_full_path}")
                
        except Exception as e:
            print(f"Error loading SPC data: {e}")
    
    def load_existing_data(self):
        """Load existing enhanced crop data"""
        try:
            existing = pd.read_csv(self.existing_csv_path)
            if not existing.empty:
                # Ensure columns match
                expected_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
                if set(existing.columns) == set(expected_cols):
                    self.enhanced_data = pd.concat([self.enhanced_data, existing], ignore_index=True)
                    print(f"Loaded existing enhanced data: {len(existing)} rows")
                else:
                    print("Warning: Existing enhanced CSV has different columns.")
        except Exception as e:
            print(f"Error loading enhanced data: {e}")
    
    def _generate_feature(self, mean, min_val, max_val, variation=0.15, dist="normal"):
        """
        Generate a single feature value with optional skew and realistic variation.
        dist can be "normal", "triangular", or "lognormal".
        """
        if dist == "normal":
            std_dev = (max_val - min_val) * variation
            val = np.random.normal(mean, std_dev)
        elif dist == "triangular":
            mode = np.clip(mean, min_val, max_val)
            val = np.random.triangular(min_val, mode, max_val)
        elif dist == "lognormal":
            sigma = variation
            val = np.random.lognormal(np.log(mean), sigma)
        else:
            val = np.random.uniform(min_val, max_val)
        
        return np.clip(val, min_val, max_val)
    
    def generate_crop_data(self, crop_name, n_samples, 
                        ph_range, ec_range, humidity_range, 
                        sunlight_range, soil_temp_range, soil_moisture_range,
                        ph_mean=None, ec_mean=None, humidity_mean=None,
                        sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        
        # Set default means if not provided
        if ph_mean is None: ph_mean = (ph_range[0] + ph_range[1]) / 2
        if ec_mean is None: ec_mean = (ec_range[0] + ec_range[1]) / 2
        if humidity_mean is None: humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if sunlight_mean is None: sunlight_mean = (sunlight_range[0] + sunlight_range[1]) / 2
        if soil_temp_mean is None: soil_temp_mean = (soil_temp_range[0] + soil_temp_range[1]) / 2
        if soil_moisture_mean is None: soil_moisture_mean = (soil_moisture_range[0] + soil_moisture_range[1]) / 2
        
        rows = []
        for i in range(n_samples):
            # Add slight random variation to means for each sample to ensure diversity
            sample_variation = 0.02  # 2% random variation in means
            
            # Generate unique means for this specific sample
            ph_sample_mean = ph_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            ec_sample_mean = ec_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            humidity_sample_mean = humidity_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            sunlight_sample_mean = sunlight_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_temp_sample_mean = soil_temp_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_moisture_sample_mean = soil_moisture_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            
            # 1. Soil pH: Normal - tightly controlled
            # Increase variation slightly and round to 1 decimal
            soil_ph = self._generate_feature(ph_sample_mean, *ph_range, variation=0.08, dist="normal")
            soil_ph = round(soil_ph, 1)
            
            # 2. EC: Lognormal 
            fertility_ec = round(self._generate_feature(ec_sample_mean, *ec_range, variation=0.05, dist="normal"), 1)
            
            # 3. Humidity: Triangular
            # humidity = round(self._generate_feature(humidity_sample_mean, *humidity_range, variation=0.12, dist="triangular"))
            humidity = round(self._generate_feature(humidity_sample_mean, *humidity_range, variation=0.12, dist="normal"))
            
            # 4. Sunlight: Add more variation and correlation
            if self.realism_mode:
                # Add temperature correlation and random factor
                temp_factor = soil_temp_sample_mean / 25
                sunlight_random = sunlight_sample_mean * (0.85 + 0.3 * np.random.random())
                sunlight_adj_mean = sunlight_random * temp_factor
                # sunlight = round(self._generate_feature(sunlight_adj_mean, *sunlight_range, variation=0.2, dist="triangular"))
                sunlight = round(self._generate_feature(sunlight_adj_mean, *sunlight_range, variation=0.2, dist="normal"))
            else:
                sunlight = round(self._generate_feature(sunlight_sample_mean, *sunlight_range, variation=0.2, dist="normal"))
            
            # 5. Soil temp: Add daily variation pattern
            # Simulate different times of day
            daily_variation = 0.05 * np.sin(i * 2 * np.pi / n_samples)  # Sinusoidal pattern
            soil_temp_mean_adj = soil_temp_sample_mean * (1 + daily_variation)
            soil_temp = self._generate_feature(soil_temp_mean_adj, *soil_temp_range, variation=0.1, dist="normal")
            soil_temp = round(soil_temp, 1)
            
            # 6. Soil moisture: Correlated with humidity
            if self.realism_mode:
                # Stronger correlation with humidity
                moisture_factor = 0.7 + 0.3 * (humidity / 100)
                soil_moisture_adj_mean = soil_moisture_sample_mean * moisture_factor
                soil_moisture = round(self._generate_feature(
                    # soil_moisture_adj_mean, *soil_moisture_range, variation=0.15, dist="triangular"
                    soil_moisture_adj_mean, *soil_moisture_range, variation=0.15, dist="normal"
                ))
            else:
                soil_moisture = round(self._generate_feature(
                    soil_moisture_sample_mean, *soil_moisture_range, variation=0.15, dist="normal"
                ))
            
            rows.append([soil_ph, fertility_ec, humidity, sunlight, soil_temp, soil_moisture, crop_name])
        
        return rows

    def add_crop_to_dataset(self, crop_name, n_samples, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """Add new crop data to enhanced dataset"""
        synthetic_rows = self.generate_crop_data(
            crop_name, n_samples, 
            ph_range, ec_range, humidity_range, 
            sunlight_range, soil_temp_range, soil_moisture_range,
            ph_mean, ec_mean, humidity_mean,
            sunlight_mean, soil_temp_mean, soil_moisture_mean
        )
        
        df_synthetic = pd.DataFrame(
            synthetic_rows, 
            columns=["soil_ph", "fertility_ec", "humidity", "sunlight", "soil_temp", "soil_moisture", "label"]
        )

        # Combine with existing enhanced data
        self.enhanced_data = pd.concat([self.enhanced_data, df_synthetic], ignore_index=True)
        print(f"Added {n_samples} synthetic samples for {crop_name} to enhanced data")
    
    def fill_crop_to_target(self, crop_name, target_count, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """Ensure a crop reaches target sample size (max 100 per crop)"""
        # Get merged data to check current counts
        merged = self.merge_all_data()
        
        # Check current count for this crop in merged data
        current_count = len(merged[merged['label'] == crop_name]) if not merged.empty else 0
        
        # Cap target count at 100
        target_count = min(target_count, 100)
        
        if current_count >= target_count:
            print(f"{crop_name} already has {current_count} samples (target capped at: {target_count})")
            return
        
        needed = target_count - current_count
        
        # Check how many samples are in SPC data for this crop
        spc_count = 0
        if self.spc_data is not None and not self.spc_data.empty and crop_name in self.spc_data['label'].values:
            spc_count = len(self.spc_data[self.spc_data['label'] == crop_name])
        
        # Check how many enhanced samples we already have for this crop
        enhanced_count = 0
        if not self.enhanced_data.empty and crop_name in self.enhanced_data['label'].values:
            enhanced_count = len(self.enhanced_data[self.enhanced_data['label'] == crop_name])
        
        print(f"{crop_name}: {spc_count} SPC samples, {enhanced_count} enhanced samples, need {needed} more to reach {target_count}")
        
        if needed > 0:
            self.add_crop_to_dataset(
                crop_name, needed, 
                ph_range, ec_range, humidity_range, 
                sunlight_range, soil_temp_range, soil_moisture_range,
                ph_mean, ec_mean, humidity_mean,
                sunlight_mean, soil_temp_mean, soil_moisture_mean
            )
    
    
    def merge_all_data(self):
        """Merge SPC data with enhanced data, limiting each crop to max 100"""
        # Clean crop names first (remove leading/trailing spaces and normalize)
        def clean_crop_name(name):
            if isinstance(name, str):
                # Remove leading/trailing spaces, convert to title case for consistency
                return name.strip().title()
            return name
        
        # Clean SPC data crop names
        if self.spc_data is not None and not self.spc_data.empty:
            self.spc_data['label'] = self.spc_data['label'].apply(clean_crop_name)
        
        # Clean enhanced data crop names
        if not self.enhanced_data.empty:
            self.enhanced_data['label'] = self.enhanced_data['label'].apply(clean_crop_name)
        
        # Start with empty dataset
        merged_data = pd.DataFrame()
        
        # 1. Add all SPC data first
        if self.spc_data is not None and not self.spc_data.empty:
            merged_data = self.spc_data.copy()
            print(f"Added {len(merged_data)} original SPC rows")
        
        # 2. Add enhanced data for crops not in SPC or to supplement existing ones
        if not self.enhanced_data.empty:
            # Get list of cleaned crops from SPC data
            spc_crops = set(self.spc_data['label'].unique()) if self.spc_data is not None else set()
            
            for crop in self.enhanced_data['label'].unique():
                enhanced_crop_data = self.enhanced_data[self.enhanced_data['label'] == crop]
                
                if crop in spc_crops:
                    # Crop exists in SPC, count how many we have already
                    current_count = len(merged_data[merged_data['label'] == crop])
                    needed = max(0, 100 - current_count)
                    
                    if needed > 0 and len(enhanced_crop_data) > 0:
                        # Add enhanced samples to reach 100
                        add_samples = min(needed, len(enhanced_crop_data))
                        # Use .iloc for safe slicing
                        merged_data = pd.concat([
                            merged_data, 
                            enhanced_crop_data.iloc[:add_samples]
                        ], ignore_index=True)
                        print(f"Added {add_samples} enhanced samples for {crop} (had {current_count} SPC samples)")
                else:
                    # New crop not in SPC, add up to 100 samples
                    if len(enhanced_crop_data) > 100:
                        merged_data = pd.concat([
                            merged_data, 
                            enhanced_crop_data.sample(n=100, random_state=42)
                        ], ignore_index=True)
                        print(f"Added 100 enhanced samples for new crop: {crop}")
                    else:
                        merged_data = pd.concat([merged_data, enhanced_crop_data], ignore_index=True)
                        print(f"Added {len(enhanced_crop_data)} enhanced samples for new crop: {crop}")
        
        # 3. Final limit check and clean any remaining duplicates
        if not merged_data.empty and 'label' in merged_data.columns:
            # Clean crop names in merged data
            merged_data['label'] = merged_data['label'].apply(clean_crop_name)
            
            # Group by cleaned crop name
            final_data = pd.DataFrame()
            unique_crops = merged_data['label'].unique()
            
            for crop in unique_crops:
                crop_data = merged_data[merged_data['label'] == crop]
                
                if len(crop_data) > 100:
                    # For crops that have SPC data, prioritize keeping SPC samples
                    if self.spc_data is not None and crop in self.spc_data['label'].values:
                        # Get SPC samples for this crop
                        spc_crop_samples = self.spc_data[self.spc_data['label'] == crop]
                        
                        # Keep all SPC samples
                        keep_data = spc_crop_samples.copy()
                        
                        # Add enhanced samples to reach 100
                        enhanced_needed = 100 - len(keep_data)
                        if enhanced_needed > 0:
                            # Get enhanced samples for this crop
                            enhanced_crop_samples = crop_data[~crop_data.index.isin(spc_crop_samples.index)] if len(spc_crop_samples) > 0 else crop_data
                            
                            if len(enhanced_crop_samples) > enhanced_needed:
                                enhanced_to_add = enhanced_crop_samples.sample(n=enhanced_needed, random_state=42)
                            else:
                                enhanced_to_add = enhanced_crop_samples
                            
                            keep_data = pd.concat([keep_data, enhanced_to_add], ignore_index=True)
                        
                        crop_data = keep_data
                    else:
                        # No SPC data, just sample 100
                        crop_data = crop_data.sample(n=100, random_state=42)
                    
                    print(f"Limited {crop} to 100 samples (was {len(merged_data[merged_data['label'] == crop])})")
                
                final_data = pd.concat([final_data, crop_data], ignore_index=True)
            
            # Remove any duplicate rows based on all columns
            final_data = final_data.drop_duplicates()
            return final_data
        
        return merged_data


    def get_crop_count(self, crop_name):
        """Count samples for crop in current merged dataset"""
        merged = self.merge_all_data()
        if len(merged) == 0 or 'label' not in merged.columns:
            return 0
        return len(merged[merged['label'] == crop_name])
    
    def save_dataset(self, output_path):
        """Save merged dataset to CSV"""
        merged = self.merge_all_data()
        
        # Only create directory if path contains directories
        dir_path = os.path.dirname(output_path)
        if dir_path:  # Only if there's actually a directory path
            os.makedirs(dir_path, exist_ok=True)
        
        merged.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Total samples: {len(merged)}")
        
        # Update the path so future saves will use the same file
        self.existing_csv_path = output_path
    
    
    def show_dataset_summary(self):
        """Show summary of dataset"""
        merged = self.merge_all_data()
        
        if len(merged) == 0:
            print("Dataset is empty")
            return
        
        # Clean crop names for consistent counting
        merged['label_clean'] = merged['label'].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
        
        print(f"Total rows: {len(merged)}")
        print(f"Unique crops: {len(merged['label_clean'].unique())}")
        
        # Show counts for each crop
        crop_counts = merged['label_clean'].value_counts()
        print("\nCrop distribution:")
        for crop, count in crop_counts.items():
            # Mark crops that came from SPC data
            if self.spc_data is not None and not self.spc_data.empty:
                # Clean SPC crop names for comparison
                spc_crops_clean = set(self.spc_data['label'].apply(
                    lambda x: x.strip().title() if isinstance(x, str) else x
                ).unique())
                is_spc_crop = crop in spc_crops_clean
                spc_marker = " (SPC)" if is_spc_crop else ""
            else:
                spc_marker = ""
            print(f"  {crop}{spc_marker}: {count} samples")
            if count > 100:
                print(f"    WARNING: {crop} has {count} samples (should be max 100)")
        
        # Show overall statistics
        print("\nFeature statistics:")
        for col in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']:
            if col in merged.columns:
                print(f"  {col}: min={merged[col].min():.1f}, "
                    f"max={merged[col].max():.1f}, "
                    f"mean={merged[col].mean():.1f}")
        
        # Show source breakdown
        print("\nData source breakdown:")
        if self.spc_data is not None and not self.spc_data.empty:
            print(f"  SPC data: {len(self.spc_data)} rows")
        if not self.enhanced_data.empty:
            print(f"  Enhanced data: {len(self.enhanced_data)} rows")


# Example usage
if __name__ == "__main__":
    output_path = "enhanced_crop_data.csv"
    
    # Initialize generator with SPC data path
    generator = CropDataGenerator(
        existing_csv_path=output_path,
        spc_data_path="SPC-soil-data2.csv",  # Path relative to dataset folder
        realism_mode=True
    )
    
    # Show initial summary (SPC data only)
    print("\n=== Initial SPC Data Summary ===")
    generator.show_dataset_summary()
 
 
    generator.fill_crop_to_target(
        crop_name="Coffee",
        target_count=100,
        ph_range=(5.8, 7.2), ph_mean=6.6,
        ec_range=(448, 478), ec_mean=460,                
        humidity_range=(79, 87), humidity_mean=83,          
        sunlight_range=(680, 1230), sunlight_mean=1150,  
        soil_temp_range=(26.3, 27.5), soil_temp_mean=27.1,        
        soil_moisture_range=(85, 99), soil_moisture_mean=92    
    )

  
   
    
 

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")