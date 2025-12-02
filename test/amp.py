import pandas as pd
import numpy as np
import os
import sys

class CropDataGenerator:
    def __init__(self, existing_csv_path=None, spc_data_path=None, realism_mode=True, boundary_flexibility=0.15):
        """
        Initialize the crop data generator
        
        Parameters:
        existing_csv_path (str): Path to the existing enhanced CSV file (optional)
        spc_data_path (str): Path to the original SPC data CSV file
        realism_mode (bool): If True, adjusts generated data to be closer to realistic agricultural ranges
        boundary_flexibility (float): How flexible boundaries are (0-1). Higher = more values outside nominal range.
        """
        self.existing_csv_path = existing_csv_path
        self.spc_data_path = spc_data_path
        self.realism_mode = realism_mode
        self.boundary_flexibility = boundary_flexibility  # New: flexibility parameter
        self.existing_data = pd.DataFrame(
            columns=['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
        )
        
        # Load SPC data first (if available)
        self.spc_data = None
        if spc_data_path:
            self.load_spc_data()
        
        # Load enhanced data if exists
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()


    
    def _generate_feature_with_flexibility(self, mean, min_val, max_val, variation=0.15, dist="normal", param_name=""):
        """
        Generate a single feature value with flexible boundaries.
        Allows values slightly outside nominal min/max based on boundary_flexibility.
        """
        # Calculate allowed range extension based on boundary_flexibility
        range_width = max_val - min_val
        extension = range_width * self.boundary_flexibility
        
        # Extended range for generation (but we'll handle acceptance differently)
        extended_min = min_val - extension
        extended_max = max_val + extension
        
        if dist == "normal":
            # Use extended range for standard deviation calculation
            std_dev = range_width * variation
            val = np.random.normal(mean, std_dev)
            
        elif dist == "triangular":
            # Use extended range for triangular distribution
            mode = np.clip(mean, extended_min, extended_max)
            val = np.random.triangular(extended_min, mode, extended_max)
            
        elif dist == "lognormal":
            sigma = variation
            # Adjust for lognormal to avoid extreme values
            adjusted_mean = np.log(mean)
            val = np.random.lognormal(adjusted_mean, sigma)
            
        else:
            # Uniform distribution across extended range
            val = np.random.uniform(extended_min, extended_max)
        
        # Apply acceptance probability for values outside nominal range
        if val < min_val or val > max_val:
            # Calculate how far outside the range
            if val < min_val:
                distance_ratio = (min_val - val) / range_width
            else:
                distance_ratio = (val - max_val) / range_width
            
            # Higher probability of accepting values slightly outside range
            # based on boundary_flexibility
            acceptance_prob = 1.0 - (distance_ratio / self.boundary_flexibility)
            acceptance_prob = max(0.0, min(1.0, acceptance_prob))
            
            # Randomly decide whether to keep this out-of-range value
            if np.random.random() < acceptance_prob:
                # Keep the out-of-range value (no clipping)
                return val
            else:
                # Clip to boundary with some randomness
                if val < min_val:
                    # Return a value near the minimum
                    return min_val + np.random.uniform(0, range_width * 0.05)
                else:
                    # Return a value near the maximum
                    return max_val - np.random.uniform(0, range_width * 0.05)
        
        # For values within nominal range, just return
        return val
    


    

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
                    self.existing_data = pd.concat([self.existing_data, existing], ignore_index=True)
                    print(f"Loaded existing enhanced data: {len(existing)} rows")
                    if 'label' in existing.columns:
                        print(f"Existing enhanced crops: {list(existing['label'].unique())}")
                else:
                    print("Warning: Existing enhanced CSV has different columns.")
        except Exception as e:
            print(f"Error loading enhanced data: {e}")


    
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
            # Add slight random variation to means for each sample
            sample_variation = 0.02
            
            # Generate unique means for this specific sample
            ph_sample_mean = ph_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            ec_sample_mean = ec_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            humidity_sample_mean = humidity_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            sunlight_sample_mean = sunlight_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_temp_sample_mean = soil_temp_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_moisture_sample_mean = soil_moisture_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            
            # 1. Soil pH: Use flexible boundaries
            soil_ph = self._generate_feature_with_flexibility(
                ph_sample_mean, *ph_range, variation=0.08, dist="normal", param_name="ph"
            )
            soil_ph = round(soil_ph, 1)
            
            # 2. EC: Use flexible boundaries
            fertility_ec = self._generate_feature_with_flexibility(
                ec_sample_mean, *ec_range, variation=0.05, dist="normal", param_name="ec"
            )
            fertility_ec = round(fertility_ec, 1)
            
            # 3. Humidity: Flexible boundaries
            humidity = self._generate_feature_with_flexibility(
                humidity_sample_mean, *humidity_range, variation=0.12, dist="triangular", param_name="humidity"
            )
            humidity = round(humidity)
            
            # 4. Sunlight: Add more variation and correlation
            if self.realism_mode:
                temp_factor = soil_temp_sample_mean / 25
                sunlight_random = sunlight_sample_mean * (0.85 + 0.3 * np.random.random())
                sunlight_adj_mean = sunlight_random * temp_factor
                sunlight = self._generate_feature_with_flexibility(
                    sunlight_adj_mean, *sunlight_range, variation=0.2, dist="triangular", param_name="sunlight"
                )
            else:
                sunlight = self._generate_feature_with_flexibility(
                    sunlight_sample_mean, *sunlight_range, variation=0.2, dist="normal", param_name="sunlight"
                )
            sunlight = round(sunlight)
            
            # 5. Soil temp: Add daily variation pattern
            daily_variation = 0.05 * np.sin(i * 2 * np.pi / n_samples)
            soil_temp_mean_adj = soil_temp_sample_mean * (1 + daily_variation)
            soil_temp = self._generate_feature_with_flexibility(
                soil_temp_mean_adj, *soil_temp_range, variation=0.1, dist="normal", param_name="soil_temp"
            )
            soil_temp = round(soil_temp, 1)
            
            # 6. Soil moisture: Correlated with humidity
            if self.realism_mode:
                moisture_factor = 0.7 + 0.3 * (humidity / 100)
                soil_moisture_adj_mean = soil_moisture_sample_mean * moisture_factor
                soil_moisture = self._generate_feature_with_flexibility(
                    soil_moisture_adj_mean, *soil_moisture_range, variation=0.15, dist="triangular", param_name="soil_moisture"
                )
            else:
                soil_moisture = self._generate_feature_with_flexibility(
                    soil_moisture_sample_mean, *soil_moisture_range, variation=0.15, dist="normal", param_name="soil_moisture"
                )
            soil_moisture = round(soil_moisture)
            
            rows.append([soil_ph, fertility_ec, humidity, sunlight, soil_temp, soil_moisture, crop_name])
        
        return rows


    def add_crop_to_dataset(self, crop_name, n_samples, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """Add new crop data to dataset"""
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

        # Combine with existing data
        self.existing_data = pd.concat([self.existing_data, df_synthetic], ignore_index=True)
    
    def fill_crop_to_target(self, crop_name, target_count, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None):
        """Ensure a crop reaches target sample size (max 100 per crop)"""
        # First, merge all data (SPC + existing enhanced)
        self.merge_all_data()
        
        # Check current count for this crop
        current_count = self.get_crop_count(crop_name)
        
        # Cap target count at 100
        target_count = min(target_count, 100)
        
        if current_count >= target_count:
            print(f"{crop_name} already has {current_count} samples (target capped at: {target_count})")
            # If we have more than target, sample down to target
            if current_count > target_count:
                self.downsample_crop(crop_name, target_count)
            return
        
        needed = target_count - current_count
        if needed > 0:
            self.add_crop_to_dataset(
                crop_name, needed, 
                ph_range, ec_range, humidity_range, 
                sunlight_range, soil_temp_range, soil_moisture_range,
                ph_mean, ec_mean, humidity_mean,
                sunlight_mean, soil_temp_mean, soil_moisture_mean
            )
    
    def downsample_crop(self, crop_name, target_count):
        """Reduce samples for a crop to target count"""
        if len(self.existing_data) == 0:
            return
        
        crop_indices = self.existing_data[self.existing_data['label'] == crop_name].index
        if len(crop_indices) > target_count:
            # Randomly select target_count samples to keep
            keep_indices = np.random.choice(crop_indices, target_count, replace=False)
            # Remove the extra samples
            remove_indices = [idx for idx in crop_indices if idx not in keep_indices]
            self.existing_data = self.existing_data.drop(remove_indices).reset_index(drop=True)
            print(f"Downsampled {crop_name} from {len(crop_indices)} to {target_count} samples")
    
    def merge_all_data(self):
        """Merge SPC data with enhanced data, limiting each crop to max 100"""
        # Create a fresh merged dataset
        merged_data = pd.DataFrame()
        
        # 1. First add ALL original SPC data
        if self.spc_data is not None and not self.spc_data.empty:
            merged_data = self.spc_data.copy()
            print(f"Added {len(merged_data)} original SPC rows")
        
        # 2. Then add enhanced synthetic data (excluding duplicates with SPC)
        if not self.existing_data.empty:
            if merged_data.empty:
                merged_data = self.existing_data.copy()
            else:
                # Only add enhanced data that doesn't duplicate SPC data
                # We'll check for near-duplicates based on feature values
                spc_crops = set(self.spc_data['label'].unique()) if self.spc_data is not None else set()
                
                for idx, row in self.existing_data.iterrows():
                    # Check if this crop exists in SPC data
                    if row['label'] in spc_crops:
                        # For crops that exist in SPC, check if this is a near-duplicate
                        is_duplicate = False
                        spc_crop_data = self.spc_data[self.spc_data['label'] == row['label']]
                        
                        # Check if any SPC row is very similar to this enhanced row
                        for _, spc_row in spc_crop_data.iterrows():
                            similarity = sum(
                                abs(row[col] - spc_row[col]) 
                                for col in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']
                                if col in row and col in spc_row
                            )
                            if similarity < 0.1:  # Threshold for considering duplicates
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            merged_data = pd.concat([merged_data, pd.DataFrame([row])], ignore_index=True)
                    else:
                        # For new crops not in SPC, always add
                        merged_data = pd.concat([merged_data, pd.DataFrame([row])], ignore_index=True)
            
            print(f"After adding enhanced data: {len(merged_data)} total rows")
        else:
            # If no enhanced data, just use SPC data
            merged_data = self.spc_data.copy() if self.spc_data is not None else pd.DataFrame()
        
        # 3. Limit each crop to max 100 samples, preserving original SPC data as much as possible
        if not merged_data.empty and 'label' in merged_data.columns:
            final_data = pd.DataFrame()
            
            for crop in merged_data['label'].unique():
                crop_data = merged_data[merged_data['label'] == crop]
                
                if len(crop_data) > 100:
                    # Separate SPC and enhanced data if possible
                    if self.spc_data is not None and crop in self.spc_data['label'].values:
                        spc_crop_rows = self.spc_data[self.spc_data['label'] == crop]
                        enhanced_crop_rows = crop_data[~crop_data.index.isin(spc_crop_rows.index)] if not spc_crop_rows.empty else crop_data
                        
                        # Keep all SPC data (up to 100)
                        keep_spc = spc_crop_rows
                        if len(keep_spc) > 100:
                            keep_spc = keep_spc.sample(n=100, random_state=42)
                        
                        # Add enhanced data to reach 100 total
                        remaining_needed = 100 - len(keep_spc)
                        if remaining_needed > 0 and not enhanced_crop_rows.empty:
                            if len(enhanced_crop_rows) > remaining_needed:
                                enhanced_sample = enhanced_crop_rows.sample(n=remaining_needed, random_state=42)
                            else:
                                enhanced_sample = enhanced_crop_rows
                            
                            crop_data = pd.concat([keep_spc, enhanced_sample], ignore_index=True)
                        else:
                            crop_data = keep_spc
                    else:
                        # If we can't identify SPC data, just sample 100 randomly
                        crop_data = crop_data.sample(n=100, random_state=42)
                    
                    print(f"Limited {crop} from {len(merged_data[merged_data['label'] == crop])} to 100 samples")
                
                final_data = pd.concat([final_data, crop_data], ignore_index=True)
            
            self.existing_data = final_data
            print(f"Final dataset after limiting: {len(self.existing_data)} rows")
        else:
            self.existing_data = merged_data



    def get_crop_count(self, crop_name):
        """Count samples for crop in merged dataset"""
        if len(self.existing_data) == 0 or 'label' not in self.existing_data.columns:
            return 0
        return len(self.existing_data[self.existing_data['label'] == crop_name])
    
    def save_dataset(self, output_path):
        """Save dataset to CSV"""
        # Ensure we have the merged data with limits applied
        self.merge_all_data()
        
        # Only create directory if path contains directories
        dir_path = os.path.dirname(output_path)
        if dir_path:  # Only if there's actually a directory path
            os.makedirs(dir_path, exist_ok=True)
        
        self.existing_data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Total samples: {len(self.existing_data)}")
        
        # Update the path so future saves will use the same file
        self.existing_csv_path = output_path
    
    def show_dataset_summary(self):
        """Show summary of dataset"""
        # Ensure we have the merged data
        self.merge_all_data()
        
        if len(self.existing_data) == 0:
            print("Dataset is empty")
            return
        
        print(f"Total rows: {len(self.existing_data)}")
        print(f"Unique crops: {len(self.existing_data['label'].unique())}")
        
        # Show counts for each crop
        crop_counts = self.existing_data['label'].value_counts()
        print("\nCrop distribution:")
        for crop, count in crop_counts.items():
            print(f"  {crop}: {count} samples")
        
        # Show overall statistics
        print("\nFeature statistics:")
        for col in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']:
            if col in self.existing_data.columns:
                print(f"  {col}: min={self.existing_data[col].min():.1f}, "
                      f"max={self.existing_data[col].max():.1f}, "
                      f"mean={self.existing_data[col].mean():.1f}")

# Example usage
if __name__ == "__main__":
    output_path = "enhanced_crop_data.csv"
    
    # Initialize generator with SPC data path
    generator = CropDataGenerator(
        existing_csv_path=output_path,
        spc_data_path="SPC-soil-data.csv",  # Path relative to dataset folder
        realism_mode=True
    )
    
    # Show initial summary (SPC data only)
    print("\n=== Initial SPC Data Summary ===")
    generator.show_dataset_summary()
 
    #(a) Ampalaya - tropical, heat-tolerant, moderate fertility
    generator.fill_crop_to_target(
        crop_name="Ampalaya",
        target_count=100,
        ph_range=(5.5, 7.2),  # Extended from 6.8 to 7.2
    ph_mean=6.3,  # Slightly higher mean
        ec_range=(523, 532), ec_mean=527,                
        humidity_range=(72, 77), humidity_mean=74,          
        sunlight_range=(2000, 5000), sunlight_mean=3200,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(87, 93), soil_moisture_mean=90   
    )


    

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")