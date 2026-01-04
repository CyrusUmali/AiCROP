import pandas as pd
import numpy as np
import os

class PureSMOTEAugmentation:
    
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
    
    def pure_smote_augmentation(self, crop_data, n_samples, k_neighbors=5):
        """
        PURE SMOTE - No constraints, no clipping, no mean adjustment
        Only interpolation between existing samples
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
            
            # Pure interpolation
            lambda_val = np.random.random()
            synthetic = sample + lambda_val * (neighbor - sample)
            
            synthetic_samples.append(synthetic)
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=self.feature_cols)
        
        # Only apply rounding (no clipping, no constraints)
        synthetic_df['soil_ph'] = synthetic_df['soil_ph'].round(1)
        synthetic_df['fertility_ec'] = synthetic_df['fertility_ec'].round(0).astype(int)
        synthetic_df['humidity'] = synthetic_df['humidity'].round(0).astype(int)
        synthetic_df['sunlight'] = synthetic_df['sunlight'].round(0).astype(int)
        synthetic_df['soil_temp'] = synthetic_df['soil_temp'].round(1)
        synthetic_df['soil_moisture'] = synthetic_df['soil_moisture'].round(0).astype(int)
        
        return synthetic_df
    
    def fill_crop_to_target(self, crop_name, target_count,  
                       method='smote', 
                       similar_crop=None):
        """
        Fill crop data to target count using PURE SMOTE
        All range and mean parameters are IGNORED - pure SMOTE only
        """
        target_count = min(target_count, 100)
        
        # REUSE SIMILAR CROP'S AUGMENTED DATA
        if similar_crop and similar_crop in self.enhanced_data['label'].values:
            similar_augmented = self.enhanced_data[
                (self.enhanced_data['label'] == similar_crop) & 
                (self.enhanced_data['source'].str.contains('augmented', na=False))
            ].copy()
            
            if len(similar_augmented) >= target_count:
                print(f"♻ {crop_name}: Reusing {similar_crop}'s augmented dataset")
                
                reused_data = similar_augmented.head(target_count).copy()
                reused_data['label'] = crop_name
                reused_data['source'] = f'reused_{similar_crop}'
                
                self.enhanced_data = pd.concat([self.enhanced_data, reused_data], 
                                              ignore_index=True)
                print(f"  ✓ Added {len(reused_data)} identical samples")
                return
        
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
                print(f"  → Using {similar_crop} as template")
                
                if self.spc_data is not None and similar_crop in self.spc_data['label'].values:
                    template_data = self.spc_data[self.spc_data['label'] == similar_crop].copy()
                    
                    if len(template_data) >= 5:
                        print(f"  → Template has {len(template_data)} samples")
                        
                        # PURE SMOTE on template data (no constraints, no ranges)
                        synthetic_df = self.pure_smote_augmentation(
                            template_data, needed
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
            return
        
        print(f"⚙ {crop_name}: {current_count} samples → using PURE SMOTE (no constraints)")
        
        # Generate synthetic data with PURE SMOTE (no ranges, no means, no constraints)
        synthetic_df = self.pure_smote_augmentation(crop_data, needed)
        
        if not synthetic_df.empty:
            synthetic_df['label'] = crop_name
            synthetic_df['source'] = f'augmented_{method}'
            self.enhanced_data = pd.concat([self.enhanced_data, synthetic_df], 
                                          ignore_index=True)
            print(f"  ✓ Added {len(synthetic_df)} synthetic samples (pure SMOTE)")

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
    generator = PureSMOTEAugmentation(
        existing_csv_path="enhanced_crop_data.csv",
        spc_data_path="SPC-soil-data.csv"
    )
    
    print("\n" + "="*60)
    print("PURE SMOTE AUGMENTATION - NO CONSTRAINTS")
    print("="*60)
    
    print("\nInitial dataset:")
    generator.show_summary()
    
    # Create Ampalaya's data with PURE SMOTE (no ranges, no means, no clipping)
    
    # generator.fill_crop_to_target(
    #     crop_name="Ipil Ipil",  
    #          target_count=100
    # )
    
    # generator.fill_crop_to_target(
    #     crop_name="Ampalaya",  
    #          target_count=100
    # )

    # # Squash reuses Ampalaya's data (identical values)
    # generator.fill_crop_to_target(
    #     crop_name="Squash",
    #     target_count=100,
    #     similar_crop="Ampalaya", 
    # )

    # # Pechay reuses Ampalaya's data (identical values)
    # generator.fill_crop_to_target(
    #     crop_name="Pechay",
    #     target_count=100,
    #     similar_crop="Ampalaya",  
    # )

    # generator.fill_crop_to_target(
    #     crop_name="Upo",
    #     target_count=100,
    #     similar_crop="Ampalaya",  
    # )

    # generator.fill_crop_to_target(
    #     crop_name="String Bean",
    #     target_count=100,
    #     similar_crop="Ampalaya",  
    # )








    generator.fill_crop_to_target(
        crop_name="Ampalaya", 
        target_count=100,
     
    ) 
  
    generator.fill_crop_to_target(
        crop_name="Squash",
        target_count=100,
     
    )

    generator.fill_crop_to_target(
        crop_name="Pechay",
        target_count=100,
      
    )
    
    generator.fill_crop_to_target(
        crop_name="Cacao",
        target_count=100,
      
    )

 
    
   
    generator.fill_crop_to_target(
        crop_name="Lanzones",
        target_count=100,
         
    )

 
 

    # Rambutan - requires high humidity, fertile soils, partial to full sun
    generator.fill_crop_to_target(
        crop_name="Rambutan",
        target_count=100,
        
    )

     
 
 
    # Calamansi - tropical, tolerates humidity, grows well in lowland Philippines
    generator.fill_crop_to_target(
        crop_name="Calamansi",
        target_count=100,
        
    )
 
  
    generator.fill_crop_to_target(
        crop_name="String Bean",
        target_count=100,
         
    )

 
 
 

  
    

 



    # === Gourds Group ===

    

 
    
    generator.fill_crop_to_target(
        crop_name="Upo",
        target_count=100,
         
    )

  
     

    


 
    generator.fill_crop_to_target(
        crop_name="Avocado",
        target_count=100,
         
    )

    generator.fill_crop_to_target(
        crop_name="Coffee",
        target_count=100,
       
    )

 


    generator.fill_crop_to_target(
        crop_name="Black Pepper",
        target_count=100,
          
    )
  
    generator.fill_crop_to_target(
        crop_name="Orchid",
        target_count=100,
         
    )

 
    generator.fill_crop_to_target(
        crop_name="Bamboo",
        target_count=100,
        
    )


    
    


    # print("Peppers: sili panigang, sili tingala")

 
    generator.fill_crop_to_target(
        crop_name="Sili Labuyo",
        target_count=100,
        
    )




  
    
    # === Root Crops Group ===
 
    #(a) Sweet Potato - drought-tolerant, prefers sandy soils, fast grower
    generator.fill_crop_to_target(
        crop_name="Sweet Potato",
        target_count=100,
        
    )
 
    # generator.fill_crop_to_target(
    #     crop_name="Kamoteng Baging",
    #     target_count=100,
    #     ph_range=(4.0, 5.5), ph_mean=4.5,
    #     ec_range=(340, 368), ec_mean=359,                 
    #     humidity_range=(68, 74), humidity_mean=70 ,
    #     sunlight_range=(14000, 17000), sunlight_mean=15839,   
    #     soil_temp_range=(25 , 35), soil_temp_mean=34.6,
    #     soil_moisture_range=(88, 99), soil_moisture_mean=95
    # )

    
    generator.fill_crop_to_target(
        crop_name="Katuray",
        target_count=100,
 
    )
 
    generator.fill_crop_to_target(
        crop_name="Kulo",
        target_count=100,
        
    )
 
 
    generator.fill_crop_to_target(
        crop_name="Lipote",
        target_count=100,
         
    )
 
    generator.fill_crop_to_target(
        crop_name="Sweet Sorghum",
        target_count=100,
       
    )
 
    # Ube (Purple Yam) - needs fertile, moist soils for tuber development
    generator.fill_crop_to_target(
        crop_name="Ube",
        target_count=100,
         
    )

    # Cassava - extreme drought tolerance, can grow in poor soils
    generator.fill_crop_to_target(
        crop_name="Cassava",
        target_count=100,
         
    )

 
 
    generator.fill_crop_to_target(
        crop_name="Banana",
        target_count=100, 
        
    )
   
 
    generator.fill_crop_to_target(
        crop_name="Jackfruit",
        target_count=100,
       
    )

    
 

    generator.fill_crop_to_target(
        crop_name="Coconut",
        target_count=100,
         
    )


     

 

    # Maize - heavy feeder, adaptable but needs consistent water during grain fill
    generator.fill_crop_to_target(
        crop_name="Maize",
        target_count=100,
         
    )

 
    # Papaya - tropical, higher temperature requirement, shallow-rooted & sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="Papaya",
        target_count=100,
       
    )

 
 
    
 

    generator.fill_crop_to_target(
        crop_name="Eggplant",
        target_count=100,
         
         
    ) 

    # Example: Crop with only 1 sample - use similar crop as template
    generator.fill_crop_to_target(
        crop_name="Forage Grass",
        target_count=100,
       
    )



 
    
 
    # Rice - semi-aquatic, flooded fields, high humidity, warm soil
    generator.fill_crop_to_target(
        crop_name="Rice",
        target_count=100,
        
    )

 


 
    generator.fill_crop_to_target(
        crop_name="Mango",
        target_count=100,
       
    )

 
  
    
    generator.fill_crop_to_target(
        crop_name="Pineapple",
        target_count=100,
         
    )

    # Gabi (Taro) - shade tolerant, high humidity & soil moisture, lower light
    generator.fill_crop_to_target(
        crop_name="Gabi",
        target_count=100,
        
    )

    # Ginger - shade/understory crop, high humidity and consistent moisture, moderate fertility
    generator.fill_crop_to_target(
        crop_name="Ginger",
        target_count=100,
       
    )
 

    generator.fill_crop_to_target(
        crop_name="Ipil Ipil",
        target_count=100,
     
    )
    
    
    print("\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    generator.show_summary()
    
    generator.save_dataset("enhanced_crop_data.csv")
    
    print("\n" + "="*60)
    print("PURE SMOTE (Chawla et al., 2002):")
    print("="*60)
    print("• Only interpolation between existing samples")
    print("• NO range constraints")
    print("• NO mean adjustments") 
    print("• NO clipping")
    print("• Preserves exact data distribution")
    print("• Generated values may extend beyond original range")
    print("• Identical data can be reused for similar crops")
    print("="*60)
    
    # Show what pure SMOTE actually generates
    if generator.enhanced_data is not None and not generator.enhanced_data.empty:
        print("\n" + "="*60)
        print("SMOTE GENERATION STATISTICS:")
        print("="*60)
        
        for crop in ["Ampalaya", "Squash", "Pechay"]:
            if crop in generator.enhanced_data['label'].values:
                crop_data = generator.enhanced_data[generator.enhanced_data['label'] == crop]
                augmented_data = crop_data[crop_data['source'].str.contains('augmented', na=False)]
                
                if not augmented_data.empty:
                    print(f"\n{crop} (Augmented):")
                    for feature in generator.feature_cols:
                        if feature in augmented_data.columns:
                            min_val = augmented_data[feature].min()
                            max_val = augmented_data[feature].max()
                            mean_val = augmented_data[feature].mean()
                            print(f"  {feature:<15} min: {min_val:>6.1f}, max: {max_val:>6.1f}, mean: {mean_val:>6.1f}")