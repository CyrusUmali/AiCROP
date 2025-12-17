import pandas as pd
import numpy as np
import os
import sys

class CropDataGenerator:
    def __init__(self, existing_csv_path=None, spc_data_path=None, realism_mode=True, 
                 auto_ratio_distribution=True, ratio_profile="balanced"):
        """
        Initialize the crop data generator
        
        Parameters:
        existing_csv_path (str): Path to the existing enhanced CSV file (optional)
        spc_data_path (str): Path to the original SPC data CSV file
        realism_mode (bool): If True, adjusts generated data to be closer to realistic agricultural ranges
        auto_ratio_distribution (bool): If True, automatically applies ratio distribution to all crops
        ratio_profile (str): Profile for ratio distribution - "balanced", "edge_heavy", or "center_heavy"
        """
        self.existing_csv_path = existing_csv_path
        self.spc_data_path = spc_data_path
        self.realism_mode = realism_mode
        self.auto_ratio_distribution = auto_ratio_distribution
        self.ratio_profile = ratio_profile
        
        # Define ratio distribution profiles
        self.ratio_profiles = {
            "balanced": {  # Default: 20% min, 60% middle, 20% max
                "min_region_ratio": 0.20,
                "mid_region_ratio": 0.60,
                "max_region_ratio": 0.20,
                "min_region_percent": 0.2,
                "max_region_percent": 0.2
            },
            "edge_heavy": {  # More data at edges: 30% min, 40% middle, 30% max
                "min_region_ratio": 0.30,
                "mid_region_ratio": 0.40,
                "max_region_ratio": 0.30,
                "min_region_percent": 0.2,
                "max_region_percent": 0.2
            },
            "center_heavy": {  # More data in center: 10% min, 80% middle, 10% max
                "min_region_ratio": 0.10,
                "mid_region_ratio": 0.80,
                "max_region_ratio": 0.10,
                "min_region_percent": 0.2,
                "max_region_percent": 0.2
            },
            "very_edge_heavy": {  # Heavy on edges: 40% min, 20% middle, 40% max
                "min_region_ratio": 0.40,
                "mid_region_ratio": 0.20,
                "max_region_ratio": 0.40,
                "min_region_percent": 0.2,
                "max_region_percent": 0.2
            }
        }
        
        # Default extreme samples configuration for all crops
        self.default_extreme_samples = {
            'pure_min_samples': 2,
            'pure_max_samples': 2,
            'mixed_extremes': {
                'count': 2,
                'profiles': [
                    {'soil_ph': 'min', 'fertility_ec': 'max'},
                    {'humidity': 'min', 'sunlight': 'max'}
                ]
            }
        }
        
        # Feature-specific adjustments (can override per feature if needed)
        self.feature_adjustments = {
            'soil_ph': {'min_region_percent': 0.15, 'max_region_percent': 0.15},  # Tighter for pH
            'fertility_ec': {'min_region_percent': 0.2, 'max_region_percent': 0.2},
            'humidity': {'min_region_percent': 0.2, 'max_region_percent': 0.2},
            'sunlight': {'min_region_percent': 0.25, 'max_region_percent': 0.25},  # Wider for sunlight
            'soil_temp': {'min_region_percent': 0.2, 'max_region_percent': 0.2},
            'soil_moisture': {'min_region_percent': 0.2, 'max_region_percent': 0.2}
        }
        
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
                expected_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
                if set(existing.columns) == set(expected_cols):
                    self.enhanced_data = pd.concat([self.enhanced_data, existing], ignore_index=True)
                    print(f"Loaded existing enhanced data: {len(existing)} rows")
                else:
                    print("Warning: Existing enhanced CSV has different columns.")
        except Exception as e:
            print(f"Error loading enhanced data: {e}")
    
    def _get_auto_ratio_distribution(self, crop_name=None):
        """
        Automatically generate ratio distribution for all features
        Can be overridden per crop if needed
        """
        if not self.auto_ratio_distribution:
            return None
        
        # Get base profile
        base_profile = self.ratio_profiles.get(self.ratio_profile, self.ratio_profiles["balanced"])
        
        # Apply to all features
        ratio_dist = {}
        for feature in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']:
            # Start with base profile
            feature_profile = base_profile.copy()
            
            # Apply feature-specific adjustments
            if feature in self.feature_adjustments:
                feature_profile.update(self.feature_adjustments[feature])
            
            # Crop-specific adjustments (example - you can expand this)
            if crop_name and crop_name.lower() in ['ampalaya', 'bitter gourd']:
                if feature == 'soil_ph':
                    # Ampalaya prefers slightly acidic soil
                    feature_profile['min_region_ratio'] = 0.30
                    feature_profile['max_region_ratio'] = 0.15
                    feature_profile['mid_region_ratio'] = 0.55
            
            ratio_dist[feature] = feature_profile
        
        return ratio_dist
    
    def _generate_feature(self, mean, min_val, max_val, variation=0.15, dist="normal", 
                         ratio_distribution=None, ensure_min_max=False):
        """
        Generate a single feature value with ratio control and optional min/max enforcement.
        """
        # If ratio distribution is specified, use it
        if ratio_distribution is not None:
            rand_val = np.random.random()
            
            # Get ratios (default to 0 if not specified)
            min_ratio = ratio_distribution.get('min_region_ratio', 0)
            max_ratio = ratio_distribution.get('max_region_ratio', 0)
            mid_ratio = ratio_distribution.get('mid_region_ratio', 0)
            
            # Normalize ratios to sum to 1
            total = min_ratio + mid_ratio + max_ratio
            if total > 0:
                min_ratio /= total
                mid_ratio /= total
                max_ratio /= total
            
            # Define region boundaries
            min_region_percent = ratio_distribution.get('min_region_percent', 0.2)
            max_region_percent = ratio_distribution.get('max_region_percent', 0.2)
            
            min_region_end = min_val + min_region_percent * (max_val - min_val)
            max_region_start = max_val - max_region_percent * (max_val - min_val)
            
            # Determine which region based on random value
            if rand_val < min_ratio:
                # Generate in min region
                val = np.random.uniform(min_val, min_region_end)
            elif rand_val < min_ratio + mid_ratio:
                # Generate in middle region
                val = np.random.uniform(min_region_end, max_region_start)
            else:
                # Generate in max region
                val = np.random.uniform(max_region_start, max_val)
            
            return np.clip(val, min_val, max_val)
        
        # Original distribution logic
        if dist == "normal":
            std_dev = (max_val - min_val) * variation
            val = np.random.normal(mean, std_dev)
        elif dist == "triangular":
            mode = np.clip(mean, min_val, max_val)
            val = np.random.triangular(min_val, mode, max_val)
        elif dist == "lognormal":
            sigma = variation
            val = np.random.lognormal(np.log(mean), sigma)
        elif dist == "uniform":
            val = np.random.uniform(min_val, max_val)
        else:
            val = np.random.uniform(min_val, max_val)
        
        return np.clip(val, min_val, max_val)
    
    def _generate_controlled_extreme_sample(self, crop_name, feature_profile, 
                                          ph_range, ec_range, humidity_range,
                                          sunlight_range, soil_temp_range, soil_moisture_range,
                                          ph_mean, ec_mean, humidity_mean,
                                          sunlight_mean, soil_temp_mean, soil_moisture_mean):
        """
        Generate a sample with controlled extreme values.
        """
        features = {}
        feature_configs = {
            'soil_ph': (ph_mean, ph_range),
            'fertility_ec': (ec_mean, ec_range),
            'humidity': (humidity_mean, humidity_range),
            'sunlight': (sunlight_mean, sunlight_range),
            'soil_temp': (soil_temp_mean, soil_temp_range),
            'soil_moisture': (soil_moisture_mean, soil_moisture_range)
        }
        
        for feat_name, (mean, rng) in feature_configs.items():
            profile = feature_profile.get(feat_name, 'normal')
            
            if profile == 'min':
                features[feat_name] = rng[0]
            elif profile == 'max':
                features[feat_name] = rng[1]
            elif profile == 'near_min':
                features[feat_name] = rng[0] + 0.05 * (rng[1] - rng[0]) * np.random.random()
            elif profile == 'near_max':
                features[feat_name] = rng[1] - 0.05 * (rng[1] - rng[0]) * np.random.random()
            else:  # 'normal'
                features[feat_name] = self._generate_feature(
                    mean, rng[0], rng[1], variation=0.1, dist="normal"
                )
        
        return [
            features['soil_ph'], features['fertility_ec'], features['humidity'],
            features['sunlight'], features['soil_temp'], features['soil_moisture'],
            crop_name
        ]
    
    def generate_crop_data(self, crop_name, n_samples, 
                          ph_range, ec_range, humidity_range, 
                          sunlight_range, soil_temp_range, soil_moisture_range,
                          ph_mean=None, ec_mean=None, humidity_mean=None,
                          sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None,
                          extreme_samples=None,
                          custom_ratio_distribution=None):
        """
        Generate crop data with automatic or custom ratio distributions.
        """
        # Set default means if not provided
        if ph_mean is None: ph_mean = (ph_range[0] + ph_range[1]) / 2
        if ec_mean is None: ec_mean = (ec_range[0] + ec_range[1]) / 2
        if humidity_mean is None: humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if sunlight_mean is None: sunlight_mean = (sunlight_range[0] + sunlight_range[1]) / 2
        if soil_temp_mean is None: soil_temp_mean = (soil_temp_range[0] + soil_temp_range[1]) / 2
        if soil_moisture_mean is None: soil_moisture_mean = (soil_moisture_range[0] + soil_moisture_range[1]) / 2
        
        # Get ratio distribution (auto if enabled, otherwise custom or none)
        if custom_ratio_distribution is not None:
            ratio_distribution = custom_ratio_distribution
        elif self.auto_ratio_distribution:
            ratio_distribution = self._get_auto_ratio_distribution(crop_name)
        else:
            ratio_distribution = None
        
        # Use default extreme samples if none provided
        if extreme_samples is None and self.default_extreme_samples:
            extreme_samples = self.default_extreme_samples
        
        rows = []
        total_extreme_samples = 0
        
        # Add controlled extreme samples if specified
        if extreme_samples:
            # Add pure minimum samples
            for _ in range(extreme_samples.get('pure_min_samples', 0)):
                rows.append([
                    ph_range[0], ec_range[0], humidity_range[0],
                    sunlight_range[0], soil_temp_range[0], soil_moisture_range[0],
                    crop_name
                ])
                total_extreme_samples += 1
            
            # Add pure maximum samples
            for _ in range(extreme_samples.get('pure_max_samples', 0)):
                rows.append([
                    ph_range[1], ec_range[1], humidity_range[1],
                    sunlight_range[1], soil_temp_range[1], soil_moisture_range[1],
                    crop_name
                ])
                total_extreme_samples += 1
            
            # Add mixed extreme samples
            if 'mixed_extremes' in extreme_samples:
                mix_config = extreme_samples['mixed_extremes']
                mix_count = mix_config.get('count', 0)
                profiles = mix_config.get('profiles', [])
                
                for i in range(mix_count):
                    if profiles:
                        profile = profiles[i % len(profiles)]
                    else:
                        profile = {}
                        for feat in ['soil_ph', 'fertility_ec', 'humidity', 
                                   'sunlight', 'soil_temp', 'soil_moisture']:
                            rand = np.random.random()
                            if rand < 0.3:
                                profile[feat] = 'min'
                            elif rand < 0.6:
                                profile[feat] = 'max'
                            else:
                                profile[feat] = 'normal'
                    
                    mixed_sample = self._generate_controlled_extreme_sample(
                        crop_name, profile,
                        ph_range, ec_range, humidity_range,
                        sunlight_range, soil_temp_range, soil_moisture_range,
                        ph_mean, ec_mean, humidity_mean,
                        sunlight_mean, soil_temp_mean, soil_moisture_mean
                    )
                    rows.append(mixed_sample)
                    total_extreme_samples += 1
        
        # Generate remaining samples
        remaining_samples = n_samples - total_extreme_samples
        
        for i in range(remaining_samples):
            sample_variation = 0.02
            
            # Generate unique means for this specific sample
            ph_sample_mean = ph_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            ec_sample_mean = ec_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            humidity_sample_mean = humidity_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            sunlight_sample_mean = sunlight_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_temp_sample_mean = soil_temp_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            soil_moisture_sample_mean = soil_moisture_mean * (1 + np.random.uniform(-sample_variation, sample_variation))
            
            # Generate each feature with ratio distribution if available
            if ratio_distribution and 'soil_ph' in ratio_distribution:
                soil_ph = self._generate_feature(
                    ph_sample_mean, *ph_range, variation=0.08, dist="normal",
                    ratio_distribution=ratio_distribution['soil_ph']
                )
            else:
                soil_ph = self._generate_feature(ph_sample_mean, *ph_range, variation=0.08, dist="normal")
            soil_ph = round(soil_ph, 1)
            
            if ratio_distribution and 'fertility_ec' in ratio_distribution:
                fertility_ec = self._generate_feature(
                    ec_sample_mean, *ec_range, variation=0.05, dist="normal",
                    ratio_distribution=ratio_distribution['fertility_ec']
                )
            else:
                fertility_ec = self._generate_feature(ec_sample_mean, *ec_range, variation=0.05, dist="normal")
            fertility_ec = int(fertility_ec + 0.5)
            
            if ratio_distribution and 'humidity' in ratio_distribution:
                humidity = self._generate_feature(
                    humidity_sample_mean, *humidity_range, variation=0.12, dist="normal",
                    ratio_distribution=ratio_distribution['humidity']
                )
            else:
                humidity = self._generate_feature(humidity_sample_mean, *humidity_range, variation=0.12, dist="normal")
            humidity = round(humidity)
            
            # Sunlight
            if self.realism_mode:
                temp_factor = soil_temp_sample_mean / 25
                sunlight_random = sunlight_sample_mean * (0.85 + 0.3 * np.random.random())
                sunlight_adj_mean = sunlight_random * temp_factor
                if ratio_distribution and 'sunlight' in ratio_distribution:
                    sunlight = self._generate_feature(
                        sunlight_adj_mean, *sunlight_range, variation=0.2, dist="normal",
                        ratio_distribution=ratio_distribution['sunlight']
                    )
                else:
                    sunlight = self._generate_feature(sunlight_adj_mean, *sunlight_range, variation=0.2, dist="normal")
            else:
                if ratio_distribution and 'sunlight' in ratio_distribution:
                    sunlight = self._generate_feature(
                        sunlight_sample_mean, *sunlight_range, variation=0.2, dist="normal",
                        ratio_distribution=ratio_distribution['sunlight']
                    )
                else:
                    sunlight = self._generate_feature(sunlight_sample_mean, *sunlight_range, variation=0.2, dist="normal")
            sunlight = round(sunlight)
            
            # Soil temperature
            daily_variation = 0.05 * np.sin(i * 2 * np.pi / remaining_samples)
            soil_temp_mean_adj = soil_temp_sample_mean * (1 + daily_variation)
            if ratio_distribution and 'soil_temp' in ratio_distribution:
                soil_temp = self._generate_feature(
                    soil_temp_mean_adj, *soil_temp_range, variation=0.1, dist="normal",
                    ratio_distribution=ratio_distribution['soil_temp']
                )
            else:
                soil_temp = self._generate_feature(soil_temp_mean_adj, *soil_temp_range, variation=0.1, dist="normal")
            soil_temp = round(soil_temp, 1)
            
            # Soil moisture
            if self.realism_mode:
                moisture_factor = 0.7 + 0.3 * (humidity / 100)
                soil_moisture_adj_mean = soil_moisture_sample_mean * moisture_factor
                if ratio_distribution and 'soil_moisture' in ratio_distribution:
                    soil_moisture = self._generate_feature(
                        soil_moisture_adj_mean, *soil_moisture_range, variation=0.15, dist="normal",
                        ratio_distribution=ratio_distribution['soil_moisture']
                    )
                else:
                    soil_moisture = self._generate_feature(
                        soil_moisture_adj_mean, *soil_moisture_range, variation=0.15, dist="normal"
                    )
            else:
                if ratio_distribution and 'soil_moisture' in ratio_distribution:
                    soil_moisture = self._generate_feature(
                        soil_moisture_sample_mean, *soil_moisture_range, variation=0.15, dist="normal",
                        ratio_distribution=ratio_distribution['soil_moisture']
                    )
                else:
                    soil_moisture = self._generate_feature(
                        soil_moisture_sample_mean, *soil_moisture_range, variation=0.15, dist="normal"
                    )
            soil_moisture = round(soil_moisture)
            
            rows.append([soil_ph, fertility_ec, humidity, sunlight, soil_temp, soil_moisture, crop_name])
        
        return rows
    
    def add_crop_to_dataset(self, crop_name, n_samples, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None,
                           extreme_samples=None,
                           custom_ratio_distribution=None):
        """Add new crop data to enhanced dataset with automatic ratio control"""
        synthetic_rows = self.generate_crop_data(
            crop_name, n_samples, 
            ph_range, ec_range, humidity_range, 
            sunlight_range, soil_temp_range, soil_moisture_range,
            ph_mean, ec_mean, humidity_mean,
            sunlight_mean, soil_temp_mean, soil_moisture_mean,
            extreme_samples,
            custom_ratio_distribution
        )
        
        df_synthetic = pd.DataFrame(
            synthetic_rows, 
            columns=["soil_ph", "fertility_ec", "humidity", "sunlight", "soil_temp", "soil_moisture", "label"]
        )

        # Combine with existing enhanced data
        self.enhanced_data = pd.concat([self.enhanced_data, df_synthetic], ignore_index=True)
        print(f"Added {n_samples} synthetic samples for {crop_name} to enhanced data")
        
        # Show distribution info if auto ratio is enabled
        if self.auto_ratio_distribution:
            profile_name = self.ratio_profile
            print(f"  Applied '{profile_name}' ratio distribution automatically")
        
        if extreme_samples:
            total_extremes = sum([v for k, v in extreme_samples.items() if isinstance(v, int)])
            if 'mixed_extremes' in extreme_samples:
                total_extremes += extreme_samples['mixed_extremes'].get('count', 0)
            print(f"  Includes {total_extremes} extreme samples")
    
    def fill_crop_to_target(self, crop_name, target_count, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None,
                           extreme_samples=None,
                           custom_ratio_distribution=None):
        """Ensure a crop reaches target sample size with automatic features"""
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
        
        # Check existing samples
        spc_count = 0
        if self.spc_data is not None and not self.spc_data.empty and crop_name in self.spc_data['label'].values:
            spc_count = len(self.spc_data[self.spc_data['label'] == crop_name])
        
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
                sunlight_mean, soil_temp_mean, soil_moisture_mean,
                extreme_samples,
                custom_ratio_distribution
            )
    
    def merge_all_data(self):
        """Merge SPC data with enhanced data, limiting each crop to max 100"""
        # Clean crop names first
        def clean_crop_name(name):
            if isinstance(name, str):
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
            spc_crops = set(self.spc_data['label'].unique()) if self.spc_data is not None else set()
            
            for crop in self.enhanced_data['label'].unique():
                enhanced_crop_data = self.enhanced_data[self.enhanced_data['label'] == crop]
                
                if crop in spc_crops:
                    current_count = len(merged_data[merged_data['label'] == crop])
                    needed = max(0, 100 - current_count)
                    
                    if needed > 0 and len(enhanced_crop_data) > 0:
                        add_samples = min(needed, len(enhanced_crop_data))
                        merged_data = pd.concat([
                            merged_data, 
                            enhanced_crop_data.iloc[:add_samples]
                        ], ignore_index=True)
                        print(f"Added {add_samples} enhanced samples for {crop} (had {current_count} SPC samples)")
                else:
                    if len(enhanced_crop_data) > 100:
                        merged_data = pd.concat([
                            merged_data, 
                            enhanced_crop_data.sample(n=100, random_state=42)
                        ], ignore_index=True)
                        print(f"Added 100 enhanced samples for new crop: {crop}")
                    else:
                        merged_data = pd.concat([merged_data, enhanced_crop_data], ignore_index=True)
                        print(f"Added {len(enhanced_crop_data)} enhanced samples for new crop: {crop}")
        
        # 3. Final limit check
        if not merged_data.empty and 'label' in merged_data.columns:
            merged_data['label'] = merged_data['label'].apply(clean_crop_name)
            
            final_data = pd.DataFrame()
            unique_crops = merged_data['label'].unique()
            
            for crop in unique_crops:
                crop_data = merged_data[merged_data['label'] == crop]
                
                if len(crop_data) > 100:
                    if self.spc_data is not None and crop in self.spc_data['label'].values:
                        spc_crop_samples = self.spc_data[self.spc_data['label'] == crop]
                        keep_data = spc_crop_samples.copy()
                        enhanced_needed = 100 - len(keep_data)
                        
                        if enhanced_needed > 0:
                            enhanced_crop_samples = crop_data[~crop_data.index.isin(spc_crop_samples.index)] if len(spc_crop_samples) > 0 else crop_data
                            if len(enhanced_crop_samples) > enhanced_needed:
                                enhanced_to_add = enhanced_crop_samples.sample(n=enhanced_needed, random_state=42)
                            else:
                                enhanced_to_add = enhanced_crop_samples
                            keep_data = pd.concat([keep_data, enhanced_to_add], ignore_index=True)
                        
                        crop_data = keep_data
                    else:
                        crop_data = crop_data.sample(n=100, random_state=42)
                    
                    print(f"Limited {crop} to 100 samples (was {len(merged_data[merged_data['label'] == crop])})")
                
                final_data = pd.concat([final_data, crop_data], ignore_index=True)
            
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
        
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        merged.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Total samples: {len(merged)}")
        
        self.existing_csv_path = output_path
    
    def show_dataset_summary(self):
        """Show summary of dataset"""
        merged = self.merge_all_data()
        
        if len(merged) == 0:
            print("Dataset is empty")
            return
        
        merged['label_clean'] = merged['label'].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
        
        print(f"Total rows: {len(merged)}")
        print(f"Unique crops: {len(merged['label_clean'].unique())}")
        
        crop_counts = merged['label_clean'].value_counts()
        print("\nCrop distribution:")
        for crop, count in crop_counts.items():
            if self.spc_data is not None and not self.spc_data.empty:
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
        
        print("\nFeature statistics:")
        for col in ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture']:
            if col in merged.columns:
                print(f"  {col}: min={merged[col].min():.1f}, "
                    f"max={merged[col].max():.1f}, "
                    f"mean={merged[col].mean():.1f}")
        
        print("\nData source breakdown:")
        if self.spc_data is not None and not self.spc_data.empty:
            print(f"  SPC data: {len(self.spc_data)} rows")
        if not self.enhanced_data.empty:
            print(f"  Enhanced data: {len(self.enhanced_data)} rows")
        
        if self.auto_ratio_distribution:
            print(f"\nAuto-ratio distribution: {self.ratio_profile} profile")


# Example usage - SIMPLIFIED!
if __name__ == "__main__":
    output_path = "enhanced_crop_data.csv"
    
    # Initialize generator with auto-ratio enabled
    generator = CropDataGenerator(
        existing_csv_path=output_path,
        spc_data_path="SPC-soil-data.csv",
        realism_mode=True,
        auto_ratio_distribution=True,  # Auto apply ratio to ALL crops
        ratio_profile="edge_heavy"      # Options: "balanced", "edge_heavy", "center_heavy", "very_edge_heavy"
    )
    
    print("\n=== Initial SPC Data Summary ===")
    generator.show_dataset_summary()
    
    # Generate Ampalaya data - NO ratio distribution needed, it's automatic!
    generator.fill_crop_to_target(
        crop_name="Ampalaya", 
        target_count=100,
        ph_range=(5.5, 7.3), ph_mean=6.5,
        ec_range=(523, 532), ec_mean=527,                
        humidity_range=(72, 77), humidity_mean=74,          
        sunlight_range=(2000, 5000), sunlight_mean=3200,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(87, 93), soil_moisture_mean=90   
    )

      
    generator.fill_crop_to_target(
        crop_name="Cacao",
        target_count=100,
        ph_range=(5.4, 6.2), ph_mean=5.8,
        ec_range=(390, 450), ec_mean=420,
        humidity_range=(80, 90), humidity_mean=85,
        sunlight_range=(740, 1060), sunlight_mean=920,  
        soil_temp_range=(25, 28), soil_temp_mean=26.9,
        soil_moisture_range=(80, 91), soil_moisture_mean=85
    )

    # Guyabano - sun-tolerant, prefers well-drained soils
    generator.fill_crop_to_target(
        crop_name="Guyabano",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(40000, 70000), sunlight_mean=55000,  # high sun
        soil_temp_range=(24, 30), soil_temp_mean=27,
        soil_moisture_range=(55, 75), soil_moisture_mean=65
    )

    
   
    generator.fill_crop_to_target(
        crop_name="Lanzones",
        target_count=100,
        ph_range=(3.2, 6.2), ph_mean=5.3,
        ec_range=(468, 950), ec_mean=750,
        humidity_range=(64, 88), humidity_mean=76,
        sunlight_range=(800, 4500), sunlight_mean=2700,   
        soil_temp_range=(25, 31), soil_temp_mean=27,
        soil_moisture_range=(82, 99), soil_moisture_mean=90
    )

    # Durian - deep-rooted, high water demand, sun-loving
    generator.fill_crop_to_target(
        crop_name="Durian",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.7,
        ec_range=(1200, 2000), ec_mean=1600,
        humidity_range=(75, 95), humidity_mean=85,
        sunlight_range=(40000, 70000), sunlight_mean=55000,  # high sun
        soil_temp_range=(24, 30), soil_temp_mean=27,
        soil_moisture_range=(65, 85), soil_moisture_mean=75
    )

 

    # Rambutan - requires high humidity, fertile soils, partial to full sun
    generator.fill_crop_to_target(
        crop_name="Rambutan",
        target_count=100,
        ph_range=(5.0, 6.7), ph_mean=5.8,
        ec_range=(418, 770), ec_mean=600,
        humidity_range=(78, 90), humidity_mean=84   ,
        sunlight_range=(770, 2300), sunlight_mean=1500,
        soil_temp_range=(26, 28.5), soil_temp_mean=27,
        soil_moisture_range=(78, 99), soil_moisture_mean=90
    )

     
 
 
    # Calamansi - tropical, tolerates humidity, grows well in lowland Philippines
    generator.fill_crop_to_target(
        crop_name="Calamansi",
        target_count=100,
        ph_range=(5.2, 6.3), ph_mean=5.7,
        ec_range=(400, 470), ec_mean=435,
        humidity_range=(75, 90), humidity_mean=83,   
        sunlight_range=(600, 2200), sunlight_mean=1400,
        soil_temp_range=(25.5, 29.3), soil_temp_mean=27.1,   
        soil_moisture_range=(80, 90), soil_moisture_mean=85
    )

    # Orange - subtropical, prefers drier air, cooler winters help fruiting
    generator.fill_crop_to_target(
        crop_name="Orange",
        target_count=100,
        ph_range=(5.5, 7.2), ph_mean=6.5,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(50, 75), humidity_mean=65,   # lower humidity tolerance
        sunlight_range=(40000, 85000), sunlight_mean=65000,
        soil_temp_range=(18, 28), soil_temp_mean=23,  # cooler compared to calamansi
        soil_moisture_range=(45, 65), soil_moisture_mean=55
    )




 

  # === Legumes Group ===

    #(a) Snap Bean - prefers cooler, moist conditions
    generator.fill_crop_to_target(
        crop_name="Snap Bean",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(500, 800), ec_mean=600,
        humidity_range=(69, 81), humidity_mean=77,
        sunlight_range=(1100, 13100), sunlight_mean=7100,  
        soil_temp_range=(23, 29), soil_temp_mean=26,         
        soil_moisture_range=(84, 94), soil_moisture_mean=89
    )

 
    generator.fill_crop_to_target(
        crop_name="String Bean",
        target_count=100,
        ph_range=(6.3, 7.5), ph_mean=6.9,
        ec_range=(500, 600), ec_mean=540,
        humidity_range=(72, 80), humidity_mean=76,         
        sunlight_range=(2000, 3000), sunlight_mean=2700,  
        soil_temp_range=(26, 32), soil_temp_mean=29,        
        soil_moisture_range=(90, 96), soil_moisture_mean=93   
    )

    # Sigarilyas (Winged Bean) - tropical, rainfall-adapted
    generator.fill_crop_to_target(
        crop_name="Sigarilyas",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1200, 2000), ec_mean=1600,
        humidity_range=(65, 85), humidity_mean=75,
        sunlight_range=(40000, 80000), sunlight_mean=60000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(55, 80), soil_moisture_mean=68  # higher moisture demand
    )

    # Mungbean - drought-tolerant, low fertility needs
    generator.fill_crop_to_target(
        crop_name="Mungbean",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,                  # less fertilizer needed
        humidity_range=(50, 75), humidity_mean=65,           # can grow in drier air
        sunlight_range=(45000, 90000), sunlight_mean=70000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(35, 65), soil_moisture_mean=50  # lowest water demand in group
    )


 


    
 
#(*) Pechay - tropical, fast growth, high fertility & water demand
    generator.fill_crop_to_target(
        crop_name="Pechay",
        target_count=100,
        ph_range=(6.0, 7.2), ph_mean=6.6,
        ec_range=(475, 595), ec_mean=535,             
        humidity_range=(68, 82), humidity_mean=75,            
        sunlight_range=(2400, 2800), sunlight_mean=2600,
        soil_temp_range=(26, 31), soil_temp_mean=28.7,          
        soil_moisture_range=(91, 97), soil_moisture_mean=94  
    )

    # Mustard - more cool-tolerant, lower fertility, tolerates drier soils
    generator.fill_crop_to_target(
        crop_name="Mustard",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(800, 1600), ec_mean=1200,                 # lower fertility need
        humidity_range=(50, 75), humidity_mean=62,          # tolerates drier air
        sunlight_range=(30000, 80000), sunlight_mean=55000, # full sun tolerant
        soil_temp_range=(15, 25), soil_temp_mean=20,        # prefers cooler soils
        soil_moisture_range=(40, 70), soil_moisture_mean=55 # tolerates drier soil
    )

 



    # === Gourds Group ===

    



    # Patola (Sponge Gourd) - high fertility, water-loving
    generator.fill_crop_to_target(
        crop_name="Patola",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,                 # higher fertility demand
        humidity_range=(60, 85), humidity_mean=75,           # prefers more humidity
        sunlight_range=(50000, 90000), sunlight_mean=70000,  # loves strong sun
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(55, 80), soil_moisture_mean=68  # consistent water needed
    )
 
    
    generator.fill_crop_to_target(
        crop_name="Upo",
        target_count=100,
        ph_range=(6.5, 7.3), ph_mean=6.9,
        ec_range=(530, 560), ec_mean=540,
        humidity_range=(72, 80), humidity_mean=76,
        sunlight_range=(2000, 3000), sunlight_mean=2700,
        soil_temp_range=(26.5, 31.5), soil_temp_mean=29.0,
        soil_moisture_range=(90, 96), soil_moisture_mean=93.0
    )

  
     

    generator.fill_crop_to_target(
        crop_name="Squash",
        target_count=100,
        ph_range=(5.8, 7.2), ph_mean=6.5,                     
        ec_range=(500, 600), ec_mean=550,                  
        humidity_range=(68, 80), humidity_mean=75,           
        sunlight_range=(2000, 10000), sunlight_mean=6000,
        soil_temp_range=(27, 32), soil_temp_mean=30,
        soil_moisture_range=(85, 94), soil_moisture_mean=90  # drought-tolerant
    )


 
    generator.fill_crop_to_target(
        crop_name="Avocado",
        target_count=100,
        ph_range=(5.4, 6.3), ph_mean=5.9,
        ec_range=(408, 462), ec_mean=435,                
        humidity_range=(78, 88), humidity_mean=83,          
        sunlight_range=(940, 1800), sunlight_mean=1350,  
        soil_temp_range=(26, 29), soil_temp_mean=27.5,        
        soil_moisture_range=(79, 89), soil_moisture_mean=85   
    )

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

 


    generator.fill_crop_to_target(
        crop_name="Black Pepper",
        target_count=100,
        ph_range=(5.7, 6.3), ph_mean=6.0,
        ec_range=(438, 443), ec_mean=442,                
        humidity_range=(83, 88), humidity_mean=85,          
        sunlight_range=(890, 920), sunlight_mean=905,  
        soil_temp_range=(26, 29), soil_temp_mean=28.7,        
        soil_moisture_range=(83, 88), soil_moisture_mean=85   
    )
  
    generator.fill_crop_to_target(
        crop_name="Orchid",
        target_count=100,
        ph_range=(5.7, 6.3), ph_mean=6.0,
        ec_range=(387, 393), ec_mean=390,                
        humidity_range=(73, 79), humidity_mean=76,          
        sunlight_range=(1600, 2000), sunlight_mean=1850,  
        soil_temp_range=(26, 28), soil_temp_mean=27.1,        
        soil_moisture_range=(76, 82), soil_moisture_mean=78   
    )


    generator.fill_crop_to_target(
        crop_name="Bamboo",
        target_count=100,
        ph_range=(4, 5.5), ph_mean=4.5,
        ec_range=(628, 637), ec_mean=632,                
        humidity_range=(60, 66), humidity_mean=62,          
        sunlight_range=(6000, 12000), sunlight_mean=8077,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(92, 99), soil_moisture_mean=95   
    )


    
    


    # print("Peppers: sili panigang, sili tingala")

    
    # === Peppers Group ===

    # Sili Panigang (Long Green Chili) - vegetable use, prefers moderate humidity and moisture
    generator.fill_crop_to_target(
        crop_name="Sili Panigang",
        target_count=100,
        ph_range=(5.4, 6.6), ph_mean=6.0,
        ec_range=(430, 570), ec_mean=500,                
        humidity_range=(50, 75), humidity_mean=62,            
        sunlight_range=(2000, 9000), sunlight_mean=5000,   
        soil_temp_range=(27.1, 31.5), soil_temp_mean=29,       
        soil_moisture_range=(81.2, 88.4), soil_moisture_mean=84 
    )

    # Sili Tingala (Bird’s Eye Chili) - hot pepper, stress-tolerant
    generator.fill_crop_to_target(
        crop_name="Sili Tingala",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(440, 560), ec_mean=500,                
        humidity_range=(50, 75), humidity_mean=62,            
        sunlight_range=(2000, 9000), sunlight_mean=5000,   
        soil_temp_range=(27.1, 31.5), soil_temp_mean=29,       
        soil_moisture_range=(81, 88), soil_moisture_mean=84 
    )
 
    generator.fill_crop_to_target(
        crop_name="Sili Labuyo",
        target_count=100,
        ph_range=(4.9, 6.1), ph_mean=5.5,
        ec_range=(450, 550), ec_mean=485,                 
        humidity_range=(72, 80), humidity_mean=76,           
        sunlight_range=(2200, 2600), sunlight_mean=2400,  
        soil_temp_range=(27, 31), soil_temp_mean=28.9,         
        soil_moisture_range=(84, 90), soil_moisture_mean=87   
    )




  
    
    # === Root Crops Group ===
 
    #(a) Sweet Potato - drought-tolerant, prefers sandy soils, fast grower
    generator.fill_crop_to_target(
        crop_name="Sweet Potato",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(400, 600), ec_mean=495,                 
        humidity_range=(78, 86), humidity_mean=82 ,
        sunlight_range=(1000, 2500), sunlight_mean=1650,   
        soil_temp_range=(25 , 30), soil_temp_mean=27.8,
        soil_moisture_range=(89, 95), soil_moisture_mean=92 
    )
 
    generator.fill_crop_to_target(
        crop_name="Kamoteng Baging",
        target_count=100,
        ph_range=(4.0, 5.5), ph_mean=4.5,
        ec_range=(340, 368), ec_mean=359,                 
        humidity_range=(68, 74), humidity_mean=70 ,
        sunlight_range=(14000, 17000), sunlight_mean=15839,   
        soil_temp_range=(25 , 35), soil_temp_mean=34.6,
        soil_moisture_range=(88, 99), soil_moisture_mean=95
    )

    
    generator.fill_crop_to_target(
        crop_name="Katuray",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6,
        ec_range=(370, 450), ec_mean=410,                 
        humidity_range=(74, 82), humidity_mean=78 ,
        sunlight_range=(2000, 2500), sunlight_mean=2250,   
        soil_temp_range=(26.5 , 31), soil_temp_mean=28.4,
        soil_moisture_range=(80, 88), soil_moisture_mean=83
    )
 
    generator.fill_crop_to_target(
        crop_name="Kulo",
        target_count=100,
        ph_range=(5.2, 6.2), ph_mean=5.7,
        ec_range=(440, 500), ec_mean=470,                 
        humidity_range=(80, 88), humidity_mean=84 ,
        sunlight_range=(830, 1280), sunlight_mean=1050,   
        soil_temp_range=(25.6 , 31), soil_temp_mean=27.4,
        soil_moisture_range=(84, 93), soil_moisture_mean=89
    )
 
 
    generator.fill_crop_to_target(
        crop_name="Lipote",
        target_count=100,
        ph_range=(5.3, 6.1), ph_mean=5.7,
        ec_range=(450, 480), ec_mean=420,                 
        humidity_range=(78, 85), humidity_mean=81 ,
        sunlight_range=(835, 1255), sunlight_mean=950 ,   
        soil_temp_range=(26.6 , 29), soil_temp_mean=27.4,
        soil_moisture_range=(78, 83), soil_moisture_mean=80
    )
 
    generator.fill_crop_to_target(
        crop_name="Sweet Sorghum",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.0,
        ec_range=(460, 600), ec_mean=515,                 
        humidity_range=(75, 86), humidity_mean=80 ,
        sunlight_range=(2000, 4000), sunlight_mean=2750,   
        soil_temp_range=(25 , 30), soil_temp_mean=28.4,
        soil_moisture_range=(88, 93), soil_moisture_mean=89  
    )
 
    # Ube (Purple Yam) - needs fertile, moist soils for tuber development
    generator.fill_crop_to_target(
        crop_name="Ube",
        target_count=100,
        ph_range=(5, 5.8), ph_mean=5.4,
        ec_range=(460, 480), ec_mean=470,                 
        humidity_range=(80, 86), humidity_mean=83,           
        sunlight_range=(1000, 2000), sunlight_mean=1300,   
        soil_temp_range=(26, 29.2), soil_temp_mean=27.9,
        soil_moisture_range=(86, 94), soil_moisture_mean=90  
    )

    # Cassava - extreme drought tolerance, can grow in poor soils
    generator.fill_crop_to_target(
        crop_name="Cassava",
        target_count=100,
        ph_range=(5.0, 6.8), ph_mean=5.8,
        ec_range=(508, 530), ec_mean=517,                  # lowest fertility requirement
        humidity_range=(85, 90), humidity_mean=88,           # drier air tolerated
        sunlight_range=(1080, 1105), sunlight_mean=1094,  # loves full sun
        soil_temp_range=(24, 34), soil_temp_mean=28,         # hotter than others
        soil_moisture_range=(90, 99), soil_moisture_mean=94  # lowest moisture need
    )

 
 
    generator.fill_crop_to_target(
        crop_name="Banana",
        target_count=100, 
        ph_range=(3.8, 7.1), ph_mean=5.0,
        ec_range=(468, 814), ec_mean=594,                  
        humidity_range=(65, 85), humidity_mean= 75,
        sunlight_range=(1280, 9437), sunlight_mean=3895,   
        soil_temp_range=(27, 30), soil_temp_mean=28.5,
        soil_moisture_range=(88, 99), soil_moisture_mean=95.8  
    )
   
 
    generator.fill_crop_to_target(
        crop_name="Jackfruit",
        target_count=100,
        ph_range=(5.5, 6.3), ph_mean=6,
        ec_range=(455, 495), ec_mean=470,                  # lower fertility need
        humidity_range=(80, 87), humidity_mean=84,
        sunlight_range=(820, 1280), sunlight_mean=1050,  # tolerates partial sun
        soil_temp_range=(26, 30), soil_temp_mean=27.4,
        soil_moisture_range=(85, 93), soil_moisture_mean=89  # moderate water demand
    )

    
 

    generator.fill_crop_to_target(
        crop_name="Coconut",
        target_count=100,
        ph_range=(5.4, 6.8), ph_mean=6.4,
        ec_range=(330, 560), ec_mean=430,
        humidity_range=(67, 86), humidity_mean=78,
        sunlight_range=(950, 13000), sunlight_mean=11591,  
        soil_temp_range=(25.8, 34), soil_temp_mean=29,
        soil_moisture_range=(84, 99), soil_moisture_mean=90   
    )


     

 

    # Maize - heavy feeder, adaptable but needs consistent water during grain fill
    generator.fill_crop_to_target(
        crop_name="Maize",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.7,
        ec_range=(460, 540), ec_mean=500,               
        humidity_range=(75, 85), humidity_mean=80,          
        sunlight_range=(1300, 3000), sunlight_mean=2200,
        soil_temp_range=(26.5, 30), soil_temp_mean=28.5,
        soil_moisture_range=(84, 93), soil_moisture_mean=87
    )

 
    # Papaya - tropical, higher temperature requirement, shallow-rooted & sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="Papaya",
        target_count=100,
        ph_range=(5.8, 6.3), ph_mean=6.0,
        ec_range=(415, 455), ec_mean=435,
        humidity_range=(76, 81), humidity_mean=79,         # more humidity-tolerant than maize
        sunlight_range=(1800, 2200), sunlight_mean=2000,
        soil_temp_range=(26, 29), soil_temp_mean=28.1,       # prefers warmer soil
        soil_moisture_range=(80, 84), soil_moisture_mean=82
    )

    # Watermelon - drought-tolerant cucurbit, wide pH range, less fertility demand
    generator.fill_crop_to_target(
        crop_name="Watermelon",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,                 # widest tolerance here
        ec_range=(800, 1600), ec_mean=1200,               # lower fertility demand
        humidity_range=(50, 70), humidity_mean=60,        # prefers drier air
        sunlight_range=(50000, 95000), sunlight_mean=75000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(40, 70), soil_moisture_mean=55  # more drought tolerant
    )


    

 
    
        # Tomato - prefers cooler root zone, high fertility, consistent moisture
    generator.fill_crop_to_target(
        crop_name="Tomato",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(2000, 3000), ec_mean=2400,            # high nutrient demand
        humidity_range=(55, 75), humidity_mean=68,      # moderate humidity
        sunlight_range=(45000, 85000), sunlight_mean=65000,
        soil_temp_range=(16, 26), soil_temp_mean=22,    # cooler root temp preferred
        soil_moisture_range=(60, 80), soil_moisture_mean=70  # steady moisture for fruit set
    )

 
  

    generator.fill_crop_to_target(
        crop_name="Eggplant",
        target_count=100,
        ph_range=(5.1, 5.9), ph_mean=5.5,
        ec_range=(400, 528), ec_mean=485,
        humidity_range=(73, 83), humidity_mean=78,       
        sunlight_range=(900, 2600), sunlight_mean=1700,
        soil_temp_range=(27.5, 31.3), soil_temp_mean=28.9,    
        soil_moisture_range=(78, 91), soil_moisture_mean=87   
    )


     #(a) Eggplant - more heat-tolerant, slightly lower fertility requirement, tolerates wider conditions
    generator.fill_crop_to_target(
        crop_name="Forage Grass",
        target_count=100,
        ph_range=(6.6, 7.3), ph_mean=7,
        ec_range=(400, 410), ec_mean=405,
        humidity_range=(76, 81), humidity_mean=79,       
        sunlight_range=(1800, 2000), sunlight_mean=1900,
        soil_temp_range=(27.5, 29.3), soil_temp_mean=28,    
        soil_moisture_range=(80, 88), soil_moisture_mean=84   
    )




    # print("Cool weather: onion, radish")


        # Onion - prefers cool to mild temps, moderate fertility, sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="Onion",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(40000, 80000), sunlight_mean=60000,
        soil_temp_range=(15, 25), soil_temp_mean=20,      # cool-moderate
        soil_moisture_range=(40, 65), soil_moisture_mean=52  # dislikes excess water
    )

    # Radish - faster growing, thrives in cooler soil, needs steadier moisture
    generator.fill_crop_to_target(
        crop_name="Radish",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1200, 2200), ec_mean=1700,              # slightly lower fertility need
        humidity_range=(50, 75), humidity_mean=62,        # tolerates a touch more humidity
        sunlight_range=(35000, 75000), sunlight_mean=55000,
        soil_temp_range=(10, 20), soil_temp_mean=15,      # cooler than onion
        soil_moisture_range=(50, 75), soil_moisture_mean=65  # steady moisture for root swelling
    )

    
     
    # Apple - temperate, requires chilling hours, cool soil temps
    generator.fill_crop_to_target(
        crop_name="Apple",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(45, 65), humidity_mean=55,        # drier temperate air
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(10, 20), soil_temp_mean=15,      # cooler soil temps
        soil_moisture_range=(45, 65), soil_moisture_mean=55
    )

    # Grapes - Mediterranean climate crop, prefers warm temps, lots of sun, tolerates drought
    generator.fill_crop_to_target(
        crop_name="Grapes",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(40, 60), humidity_mean=50,        # needs drier climate to avoid rot
        sunlight_range=(55000, 100000), sunlight_mean=80000,
        soil_temp_range=(18, 28), soil_temp_mean=23,
        soil_moisture_range=(40, 60), soil_moisture_mean=50  # drought tolerant
    )
 
    # Rice - semi-aquatic, flooded fields, high humidity, warm soil
    generator.fill_crop_to_target(
        crop_name="Rice",
        target_count=100,
        ph_range=(6.8, 7.3), ph_mean=7,
        ec_range=(530, 570), ec_mean=550,
        humidity_range=(70, 79), humidity_mean=75,        # high humidity crop
        sunlight_range=(2400, 2700), sunlight_mean=2500,
        soil_temp_range=(26, 31), soil_temp_mean=29.2,
        soil_moisture_range=(92, 98), soil_moisture_mean=95  # flooded/paddy conditions
    )

    # Oyster Mushroom - shade crop, very high humidity, moderate temps
    generator.fill_crop_to_target(
        crop_name="Oyster Mushroom",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(500, 1500), ec_mean=1000,
        humidity_range=(85, 95), humidity_mean=90,        # nearly saturated air
        sunlight_range=(2000, 10000), sunlight_mean=6000, # low light / shade
        soil_temp_range=(18, 25), soil_temp_mean=22,
        soil_moisture_range=(65, 90), soil_moisture_mean=78
    )




 
    generator.fill_crop_to_target(
        crop_name="Mango",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,
        ec_range=(380, 530), ec_mean=460,           
        humidity_range=(76, 91), humidity_mean=85,    
        sunlight_range=(950, 1150), sunlight_mean=1050,
        soil_temp_range=(26, 31), soil_temp_mean=29.5,  
        soil_moisture_range=(78, 99), soil_moisture_mean=88   
    )

    # Okra - heat-loving, full-sun annual; tolerates drier spells but performs with moderate fertility
    generator.fill_crop_to_target(
        crop_name="Okra",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1200, 2200), ec_mean=1700,           # moderate-to-high fertility
        humidity_range=(60, 85), humidity_mean=72,
        sunlight_range=(60000, 100000), sunlight_mean=80000,
        soil_temp_range=(22, 34), soil_temp_mean=28,
        soil_moisture_range=(40, 75), soil_moisture_mean=55  # tolerates some dryness
    )

  
    
    generator.fill_crop_to_target(
        crop_name="Pineapple",
        target_count=100,
        ph_range=(4.5, 6.0), ph_mean=5.3,
        ec_range=(430, 750), ec_mean=624,              
        humidity_range=(70, 80), humidity_mean=75,
        sunlight_range=(2500, 45000), sunlight_mean=20000,
        soil_temp_range=(27, 34.5), soil_temp_mean=31,
        soil_moisture_range=(85, 99), soil_moisture_mean=92 
    )

    # Gabi (Taro) - shade tolerant, high humidity & soil moisture, lower light
    generator.fill_crop_to_target(
        crop_name="Gabi",
        target_count=100,
        ph_range=(6.5, 7.2), ph_mean=6.8,
        ec_range=(458, 470), ec_mean=464,
        humidity_range=(80, 86), humidity_mean=83,     # very humid conditions
        sunlight_range=(1100, 10000), sunlight_mean=5500,  # shade to partial shade
        soil_temp_range=(26, 27.8), soil_temp_mean=26,
        soil_moisture_range=(90, 96), soil_moisture_mean=93  # likes saturated/very moist soils
    )

    # Ginger - shade/understory crop, high humidity and consistent moisture, moderate fertility
    generator.fill_crop_to_target(
        crop_name="Ginger",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(453, 470), ec_mean=460,
        humidity_range=(77, 86), humidity_mean=82,   
        sunlight_range=(1130, 1810), sunlight_mean=1500,   # low light / filtered shade
        soil_temp_range=(26, 29), soil_temp_mean=28,
        soil_moisture_range=(84, 91), soil_moisture_mean=87  # consistently moist but not waterlogged
    )

    generator.fill_crop_to_target(
        crop_name="Ipil Ipil",
        target_count=100,
        ph_range=(6.2, 7.2), ph_mean=6.8,
        ec_range=(398, 412), ec_mean=405,
        humidity_range=(77, 81), humidity_mean=79,   
        sunlight_range=(1800, 2000), sunlight_mean=1900,   # low light / filtered shade
        soil_temp_range=(26, 29), soil_temp_mean=28,
        soil_moisture_range=(82, 87), soil_moisture_mean=84  # consistently moist but not waterlogged
    )

    
   
    
 

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")