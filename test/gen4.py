import pandas as pd
import numpy as np
import os
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

class DefensibleCropDataAugmentation:
    """
    Defensible data augmentation for agricultural datasets using established techniques:
    1. SMOTE (Synthetic Minority Over-sampling Technique)
    2. Gaussian Copula-based augmentation
    3. Feature-correlation preserving noise injection
    4. Domain-constrained perturbations
    """
    
    def __init__(self, existing_csv_path=None, spc_data_path=None):
        self.existing_csv_path = existing_csv_path
        self.spc_data_path = spc_data_path
        self.spc_data = None
        self.enhanced_data = pd.DataFrame()
        self.feature_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 
                            'soil_temp', 'soil_moisture']
        
        if spc_data_path:
            self.load_spc_data()
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
    
    def load_spc_data(self):
        """Load original SPC soil data"""
        try:
            if os.path.basename(os.getcwd()) == 'test':
                parent_dir = os.path.dirname(os.getcwd())
                spc_full_path = os.path.join(parent_dir, 'dataset', self.spc_data_path)
            else:
                spc_full_path = os.path.join('dataset', self.spc_data_path)
            
            if os.path.exists(spc_full_path):
                spc_data = pd.read_csv(spc_full_path)
                
                # Auto-map columns
                column_mapping = {}
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
                
                if column_mapping:
                    spc_data = spc_data.rename(columns=column_mapping)
                    expected_cols = self.feature_cols + ['label']
                    available_cols = [col for col in expected_cols if col in spc_data.columns]
                    
                    if 'label' in available_cols:
                        spc_data = spc_data[available_cols]
                        spc_data['label'] = spc_data['label'].str.strip().str.title()
                        self.spc_data = spc_data
                        print(f"✓ Loaded SPC data: {len(spc_data)} rows")
                        print(f"  Crops: {list(spc_data['label'].unique())}")
        except Exception as e:
            print(f"Error loading SPC data: {e}")
    
    def load_existing_data(self):
        """Load existing enhanced data"""
        try:
            existing = pd.read_csv(self.existing_csv_path)
            if not existing.empty:
                existing['label'] = existing['label'].str.strip().str.title()
                self.enhanced_data = existing
                print(f"✓ Loaded enhanced data: {len(existing)} rows")
        except Exception as e:
            print(f"Error loading enhanced data: {e}")
    
    def smote_augmentation(self, crop_data, n_samples, k_neighbors=5):
        """
        SMOTE (Synthetic Minority Over-sampling Technique)
        Citation: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
        
        Generates synthetic samples by interpolating between existing samples.
        """
        if len(crop_data) < 2:
            print(f"Warning: Need at least 2 samples for SMOTE, found {len(crop_data)}")
            return pd.DataFrame()
        
        X = crop_data[self.feature_cols].values
        k = min(k_neighbors, len(crop_data) - 1)
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Randomly select a sample
            idx = np.random.randint(0, len(X))
            sample = X[idx]
            
            # Find k nearest neighbors
            distances = np.sqrt(np.sum((X - sample) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[1:k+1]
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(nearest_indices)
            neighbor = X[neighbor_idx]
            
            # Generate synthetic sample by interpolation
            lambda_val = np.random.random()
            synthetic = sample + lambda_val * (neighbor - sample)
            
            synthetic_samples.append(synthetic)
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_cols)
        return synthetic_df
    
    def gaussian_copula_augmentation(self, crop_data, n_samples):
        """
        Gaussian Copula-based augmentation
        Preserves marginal distributions and correlation structure
        
        This method:
        1. Estimates correlation structure from real data
        2. Samples from multivariate Gaussian with same correlations
        3. Transforms back to match original marginal distributions
        """
        if len(crop_data) < 5:
            print(f"Warning: Need at least 5 samples for copula, found {len(crop_data)}")
            return pd.DataFrame()
        
        X = crop_data[self.feature_cols].values
        
        # Estimate rank-based correlation (Spearman)
        from scipy.stats import spearmanr
        ranked_data = np.zeros_like(X)
        for i in range(X.shape[1]):
            ranked_data[:, i] = np.argsort(np.argsort(X[:, i])) / (len(X) - 1)
        
        # Convert ranks to Gaussian using inverse CDF
        from scipy.stats import norm
        gaussian_data = norm.ppf(np.clip(ranked_data, 0.01, 0.99))
        
        # Estimate covariance
        cov_matrix = np.cov(gaussian_data.T)
        mean = np.mean(gaussian_data, axis=0)
        
        # Generate synthetic samples in Gaussian space
        synthetic_gaussian = np.random.multivariate_normal(mean, cov_matrix, n_samples)
        
        # Transform back to uniform
        synthetic_uniform = norm.cdf(synthetic_gaussian)
        
        # Transform to match original marginals using quantiles
        synthetic_samples = np.zeros_like(synthetic_uniform)
        for i in range(X.shape[1]):
            # Use empirical quantiles
            sorted_vals = np.sort(X[:, i])
            indices = (synthetic_uniform[:, i] * (len(sorted_vals) - 1)).astype(int)
            indices = np.clip(indices, 0, len(sorted_vals) - 1)
            synthetic_samples[:, i] = sorted_vals[indices]
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_cols)
        return synthetic_df
    
    def noise_injection_augmentation(self, crop_data, n_samples, noise_level=0.05):
        """
        Correlation-preserving noise injection
        
        Adds small amounts of noise while preserving feature correlations
        Noise level is relative to feature standard deviation
        """
        if len(crop_data) == 0:
            return pd.DataFrame()
        
        X = crop_data[self.feature_cols].values
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X.T)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        # Scale noise by feature standard deviations
        noise_cov = cov_matrix * (noise_level ** 2)
        
        synthetic_samples = []
        for _ in range(n_samples):
            # Select random base sample
            base_idx = np.random.randint(0, len(X))
            base_sample = X[base_idx]
            
            # Add correlated noise
            noise = np.random.multivariate_normal(np.zeros(len(mean)), noise_cov)
            synthetic = base_sample + noise
            
            synthetic_samples.append(synthetic)
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_cols)
        return synthetic_df
    
    def domain_constrained_augmentation(self, crop_data, n_samples, 
                                       constraints=None, method='smote'):
        """
        Applies augmentation with domain-specific constraints
        
        Parameters:
        - constraints: dict with 'min' and 'max' for each feature
        - method: 'smote', 'copula', or 'noise'
        """
        # Generate synthetic data
        if method == 'smote':
            synthetic_df = self.smote_augmentation(crop_data, n_samples)
        elif method == 'copula':
            synthetic_df = self.gaussian_copula_augmentation(crop_data, n_samples)
        elif method == 'noise':
            synthetic_df = self.noise_injection_augmentation(crop_data, n_samples)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if synthetic_df.empty:
            return synthetic_df
        
        # Apply constraints if provided
        if constraints:
            for feature, bounds in constraints.items():
                if feature in synthetic_df.columns:
                    synthetic_df[feature] = np.clip(
                        synthetic_df[feature],
                        bounds['min'],
                        bounds['max']
                    )
        
        # Apply realistic rounding
        synthetic_df['soil_ph'] = synthetic_df['soil_ph'].round(1)
        synthetic_df['fertility_ec'] = synthetic_df['fertility_ec'].round(0).astype(int)
        synthetic_df['humidity'] = synthetic_df['humidity'].round(0).astype(int)
        synthetic_df['sunlight'] = synthetic_df['sunlight'].round(0).astype(int)
        synthetic_df['soil_temp'] = synthetic_df['soil_temp'].round(1)
        synthetic_df['soil_moisture'] = synthetic_df['soil_moisture'].round(0).astype(int)
        
        return synthetic_df
    
    def fill_crop_to_target(self, crop_name, target_count, 
                           ph_range, ec_range, humidity_range, 
                           sunlight_range, soil_temp_range, soil_moisture_range,
                           ph_mean=None, ec_mean=None, humidity_mean=None,
                           sunlight_mean=None, soil_temp_mean=None, soil_moisture_mean=None,
                           method='smote', 
                           similar_crop=None):
        """
        Fill crop data to target count using defensible augmentation
        
        Keeps the same interface as your original code!
        
        Parameters match your original structure, plus:
        - method: 'smote', 'copula', 'noise', or 'mixed'
        - similar_crop: For crops with <2 samples, use this similar crop as template
        
        Example for crops with only 1 sample:
            generator.fill_crop_to_target(
                crop_name="Rare_Crop",
                similar_crop="Common_Crop",  # Use this as pattern
                target_count=50,
                ph_range=(5.5, 6.5), ph_mean=6.0,
                ...
            )
        """
        target_count = min(target_count, 100)
        
        # Get current data for this crop
        if self.spc_data is not None and crop_name in self.spc_data['label'].values:
            crop_data = self.spc_data[self.spc_data['label'] == crop_name].copy()
        else:
            crop_data = pd.DataFrame()
        
        current_count = len(crop_data)
        
        if current_count >= target_count:
            print(f"✓ {crop_name}: Already has {current_count} samples (target: {target_count})")
            return
        
        needed = target_count - current_count
        
        # Handle crops with insufficient samples
        if current_count < 2:
            if similar_crop:
                print(f"⚠ {crop_name}: Only {current_count} sample(s)")
                print(f"  → Using {similar_crop} as template with custom constraints")
                
                # Get similar crop data
                if self.spc_data is not None and similar_crop in self.spc_data['label'].values:
                    template_data = self.spc_data[self.spc_data['label'] == similar_crop].copy()
                    
                    if len(template_data) >= 5:
                        # Use template crop's correlation structure
                        print(f"  → Template has {len(template_data)} samples")
                        
                        # Build constraints from ranges
                        constraints = {
                            'soil_ph': {'min': ph_range[0], 'max': ph_range[1]},
                            'fertility_ec': {'min': ec_range[0], 'max': ec_range[1]},
                            'humidity': {'min': humidity_range[0], 'max': humidity_range[1]},
                            'sunlight': {'min': sunlight_range[0], 'max': sunlight_range[1]},
                            'soil_temp': {'min': soil_temp_range[0], 'max': soil_temp_range[1]},
                            'soil_moisture': {'min': soil_moisture_range[0], 'max': soil_moisture_range[1]}
                        }
                        
                        # Generate using template's structure but our constraints
                        synthetic_df = self._augment_with_template(
                            template_data, needed, constraints, method='smote'
                        )
                        
                        if not synthetic_df.empty:
                            synthetic_df['label'] = crop_name
                            synthetic_df['source'] = f'template_{similar_crop}'
                            self.enhanced_data = pd.concat([self.enhanced_data, synthetic_df], 
                                                          ignore_index=True)
                            print(f"  ✓ Added {len(synthetic_df)} samples based on {similar_crop}")
                        return
                    else:
                        print(f"  ✗ Template {similar_crop} has only {len(template_data)} samples (need 5+)")
                else:
                    print(f"  ✗ Template crop '{similar_crop}' not found in dataset")
            
            print(f"✗ {crop_name}: Only {current_count} sample(s), need at least 2 for augmentation")
            print(f"  → Options:")
            print(f"     1. Collect more real samples (recommended)")
            print(f"     2. Use similar_crop='CropName' parameter")
            print(f"     3. Skip this crop for now")
            return
        
        # Auto-select method based on sample count
        if current_count < 5:
            method = 'smote'
            print(f"⚙ {crop_name}: {current_count} samples → using SMOTE (small dataset)")
        elif current_count < 10:
            method = 'smote' if method == 'mixed' else method
            print(f"⚙ {crop_name}: {current_count} samples → using {method}")
        else:
            print(f"⚙ {crop_name}: {current_count} samples → using {method}")
        
        # Build constraints from ranges
        constraints = {
            'soil_ph': {'min': ph_range[0], 'max': ph_range[1]},
            'fertility_ec': {'min': ec_range[0], 'max': ec_range[1]},
            'humidity': {'min': humidity_range[0], 'max': humidity_range[1]},
            'sunlight': {'min': sunlight_range[0], 'max': sunlight_range[1]},
            'soil_temp': {'min': soil_temp_range[0], 'max': soil_temp_range[1]},
            'soil_moisture': {'min': soil_moisture_range[0], 'max': soil_moisture_range[1]}
        }
        
        # Generate synthetic data
        if method == 'mixed':
            n_per_method = needed // 3
            synthetic_smote = self.domain_constrained_augmentation(
                crop_data, n_per_method, constraints, 'smote'
            )
            synthetic_copula = self.domain_constrained_augmentation(
                crop_data, n_per_method, constraints, 'copula'
            )
            synthetic_noise = self.domain_constrained_augmentation(
                crop_data, needed - 2*n_per_method, constraints, 'noise'
            )
            synthetic_df = pd.concat([synthetic_smote, synthetic_copula, synthetic_noise], 
                                    ignore_index=True)
        else:
            synthetic_df = self.domain_constrained_augmentation(
                crop_data, needed, constraints, method
            )
        
        if not synthetic_df.empty:
            synthetic_df['label'] = crop_name
            synthetic_df['source'] = f'augmented_{method}'
            self.enhanced_data = pd.concat([self.enhanced_data, synthetic_df], 
                                          ignore_index=True)
            print(f"  ✓ Added {len(synthetic_df)} synthetic samples")
    
    def _augment_with_template(self, template_data, n_samples, constraints, method='smote'):
        """
        Generate synthetic data using a template crop's correlation structure
        but constrained to target crop's ranges
        """
        # Generate samples using template
        if method == 'smote':
            synthetic_df = self.smote_augmentation(template_data, n_samples)
        elif method == 'copula':
            synthetic_df = self.gaussian_copula_augmentation(template_data, n_samples)
        else:
            synthetic_df = self.noise_injection_augmentation(template_data, n_samples)
        
        if synthetic_df.empty:
            return synthetic_df
        
        # Scale to target constraints while preserving relative relationships
        for feature in self.feature_cols:
            if feature in synthetic_df.columns and feature in constraints:
                # Get current range
                current_min = synthetic_df[feature].min()
                current_max = synthetic_df[feature].max()
                current_range = current_max - current_min
                
                # Target range
                target_min = constraints[feature]['min']
                target_max = constraints[feature]['max']
                target_range = target_max - target_min
                
                # Scale and shift
                if current_range > 0:
                    normalized = (synthetic_df[feature] - current_min) / current_range
                    synthetic_df[feature] = target_min + normalized * target_range
                else:
                    # All values the same, use target mean
                    synthetic_df[feature] = (target_min + target_max) / 2
        
        # Apply rounding
        synthetic_df['soil_ph'] = synthetic_df['soil_ph'].round(1)
        synthetic_df['fertility_ec'] = synthetic_df['fertility_ec'].round(0).astype(int)
        synthetic_df['humidity'] = synthetic_df['humidity'].round(0).astype(int)
        synthetic_df['sunlight'] = synthetic_df['sunlight'].round(0).astype(int)
        synthetic_df['soil_temp'] = synthetic_df['soil_temp'].round(1)
        synthetic_df['soil_moisture'] = synthetic_df['soil_moisture'].round(0).astype(int)
        
        return synthetic_df
    
    def merge_all_data(self):
        """Merge SPC and enhanced data, limiting to 100 per crop"""
        merged_data = pd.DataFrame()
        
        # Add SPC data
        if self.spc_data is not None and not self.spc_data.empty:
            spc_copy = self.spc_data.copy()
            spc_copy['source'] = 'original'
            merged_data = spc_copy
        
        # Add enhanced data
        if not self.enhanced_data.empty:
            for crop in self.enhanced_data['label'].unique():
                enhanced_crop = self.enhanced_data[self.enhanced_data['label'] == crop]
                current_count = len(merged_data[merged_data['label'] == crop]) if not merged_data.empty else 0
                needed = max(0, 100 - current_count)
                
                if needed > 0:
                    add_samples = min(needed, len(enhanced_crop))
                    merged_data = pd.concat([merged_data, enhanced_crop.iloc[:add_samples]], 
                                          ignore_index=True)
        
        return merged_data
    
    def save_dataset(self, output_path):
        """Save merged dataset"""
        merged = self.merge_all_data()
        
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        merged.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"  Total samples: {len(merged)}")
    
    def show_summary(self):
        """Show dataset summary with augmentation statistics"""
        merged = self.merge_all_data()
        
        if merged.empty:
            print("Dataset is empty")
            return
        
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(merged)}")
        print(f"Unique crops: {len(merged['label'].unique())}")
        
        print(f"\n{'Crop':<15} {'Total':<8} {'Original':<10} {'Augmented':<10}")
        print(f"{'-'*50}")
        
        for crop in sorted(merged['label'].unique()):
            crop_data = merged[merged['label'] == crop]
            total = len(crop_data)
            
            if 'source' in crop_data.columns:
                original = len(crop_data[crop_data['source'] == 'original'])
                augmented = total - original
            else:
                original = total
                augmented = 0
            
            print(f"{crop:<15} {total:<8} {original:<10} {augmented:<10}")


# Example usage
if __name__ == "__main__":
    # Initialize generator (same as your original!)
    generator = DefensibleCropDataAugmentation(
        existing_csv_path="enhanced_crop_data.csv",
        spc_data_path="SPC-soil-data.csv"
    )
    
    print("\n" + "="*60)
    print("DEFENSIBLE DATA AUGMENTATION FOR AGRICULTURAL DATASETS")
    print("="*60)
    
    # Show initial state
    print("\nInitial dataset:")
    generator.show_summary()
    
    # Use your EXACT original syntax! Just works better under the hood
    generator.fill_crop_to_target(
        crop_name="Ampalaya", 
        target_count=100,
        ph_range=(5.5, 7.3), ph_mean=6.5,
        ec_range=(523, 532), ec_mean=527,                
        humidity_range=(72, 77), humidity_mean=74,          
        sunlight_range=(2000, 5000), sunlight_mean=3200,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(87, 93), soil_moisture_mean=90,
        method='smote'  # Optional: specify method, or let it auto-select
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

 



    # === Gourds Group ===

    

 
    
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
        soil_moisture_range=(76, 82), soil_moisture_mean=78 ,
        method= 'smote' 
    )


    generator.fill_crop_to_target(
        crop_name="Bamboo",
        target_count=100,
        ph_range=(4, 5.5), ph_mean=4.5,
        ec_range=(628, 637), ec_mean=632,                
        humidity_range=(60, 66), humidity_mean=62,          
        sunlight_range=(6000, 12000), sunlight_mean=8077,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(92, 99), soil_moisture_mean=95,
        method= 'smote'
    )


    
    


    # print("Peppers: sili panigang, sili tingala")

 
    generator.fill_crop_to_target(
        crop_name="Sili Labuyo",
        target_count=100,
        ph_range=(4.9, 6.1), ph_mean=5.5,
        ec_range=(450, 550), ec_mean=485,                 
        humidity_range=(72, 80), humidity_mean=76,           
        sunlight_range=(2200, 2600), sunlight_mean=2400,  
        soil_temp_range=(27, 31), soil_temp_mean=28.9,         
        soil_moisture_range=(84, 90), soil_moisture_mean=87  ,
        method='smote'
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
        soil_moisture_range=(89, 95), soil_moisture_mean=92,
        # method='smote'
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
        soil_moisture_range=(86, 94), soil_moisture_mean=90 , 
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
        soil_moisture_range=(90, 99), soil_moisture_mean=94,
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
        soil_moisture_range=(85, 93), soil_moisture_mean=89, 
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

 
 
    
 

    generator.fill_crop_to_target(
        crop_name="Eggplant",
        target_count=100,
        ph_range=(5.1, 5.9), ph_mean=5.5,
        ec_range=(400, 528), ec_mean=485,
        humidity_range=(73, 83), humidity_mean=78,       
        sunlight_range=(900, 2600), sunlight_mean=1700,
        soil_temp_range=(27.5, 31.3), soil_temp_mean=28.9,    
        soil_moisture_range=(78, 91), soil_moisture_mean=87
        # method not specified = auto-selects based on sample count
    ) 

    # Example: Crop with only 1 sample - use similar crop as template
    generator.fill_crop_to_target(
        crop_name="Forage Grass",
        target_count=100,
        ph_range=(5.0, 7.3), ph_mean=7,
        ec_range=(395, 580), ec_mean=405,
        humidity_range=(72, 87), humidity_mean=79,       
        sunlight_range=(1300, 3000), sunlight_mean=1900,
        soil_temp_range=(26, 30.3), soil_temp_mean=28,    
        soil_moisture_range=(80, 98), soil_moisture_mean=84   
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
        soil_moisture_range=(92, 98), soil_moisture_mean=95,
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
    
    # Show final summary
    print("\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    generator.show_summary()
    
    # Save results
    generator.save_dataset("enhanced_crop_data.csv")
    
    print("\n" + "="*60)
    print("AUGMENTATION METHODS USED:")
    print("="*60)
    print("• SMOTE: Interpolates between existing samples")
    print("• Gaussian Copula: Preserves correlation structure")
    print("• Noise Injection: Adds small correlated perturbations")
    print("• Mixed: Combines all methods for diversity")
    print("\n✓ All methods respect your domain constraints")
    print("✓ Preserves statistical properties of original data")
    print("✓ Every sample tagged with its source for transparency")
    print("="*60)