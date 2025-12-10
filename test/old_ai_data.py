import pandas as pd
import numpy as np
import os

class CropDataGenerator:
    def __init__(self, existing_csv_path=None, realism_mode=True):
        """
        Initialize the crop data generator
        
        Parameters:
        existing_csv_path (str): Path to the existing CSV file (optional)
        realism_mode (bool): If True, adjusts generated data to be closer to realistic agricultural ranges
        """
        self.existing_csv_path = existing_csv_path
        self.realism_mode = realism_mode
        self.existing_data = pd.DataFrame(
            columns=['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
        )
        
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing crop data"""
        try:
            existing = pd.read_csv(self.existing_csv_path)
            if not existing.empty:
                # Ensure columns match
                expected_cols = ['soil_ph', 'fertility_ec', 'humidity', 'sunlight', 'soil_temp', 'soil_moisture', 'label']
                if set(existing.columns) == set(expected_cols):
                    self.existing_data = pd.concat([self.existing_data, existing], ignore_index=True)
                    print(f"Loaded existing data: {len(existing)} rows")
                    print(f"Existing crops: {list(existing['label'].unique())}")
                else:
                    print("Warning: Existing CSV has different columns. Starting with empty dataset.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _generate_feature(self, mean, min_val, max_val, variation=0.15, dist="normal"):
        """
        Generate a single feature value with optional skew and realistic variation.
        dist can be "normal", "triangular", or "lognormal".
        """
        if dist == "normal":
            std_dev = (max_val - min_val) * variation
            val = np.random.normal(mean, std_dev)
        elif dist == "triangular":
            val = np.random.triangular(min_val, mean, max_val)
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
        """
        Generate synthetic crop data with optional realism adjustments.
        """
        # Set default means if not provided
        if ph_mean is None: ph_mean = (ph_range[0] + ph_range[1]) / 2
        if ec_mean is None: ec_mean = (ec_range[0] + ec_range[1]) / 2
        if humidity_mean is None: humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if sunlight_mean is None: sunlight_mean = (sunlight_range[0] + sunlight_range[1]) / 2
        if soil_temp_mean is None: soil_temp_mean = (soil_temp_range[0] + soil_temp_range[1]) / 2
        if soil_moisture_mean is None: soil_moisture_mean = (soil_moisture_range[0] + soil_moisture_range[1]) / 2
        
        rows = []
        for _ in range(n_samples):
            # Soil pH: narrow variation
            soil_ph = round(self._generate_feature(ph_mean, *ph_range, variation=0.05, dist="normal"), 1)
            
            # Fertility (EC): moderate variation
            fertility_ec = round(self._generate_feature(ec_mean, *ec_range, variation=0.12, dist="normal"))
            
            # Humidity: moderate variation
            humidity = round(self._generate_feature(humidity_mean, *humidity_range, variation=0.08, dist="normal"))
            
            # Sunlight: correlated with temperature in realism mode
            if self.realism_mode:
                # Higher sunlight often correlates with higher soil temperature
                sunlight_adj_mean = sunlight_mean * (soil_temp_mean / 25)  # Adjust based on temperature
                sunlight = round(self._generate_feature(sunlight_adj_mean, *sunlight_range, variation=0.15, dist="normal"))
            else:
                sunlight = round(self._generate_feature(sunlight_mean, *sunlight_range, variation=0.15, dist="normal"))
            
            # Soil temperature: moderate variation
            soil_temp = round(self._generate_feature(soil_temp_mean, *soil_temp_range, variation=0.08, dist="normal"), 1)
            
            # Soil moisture: correlated with humidity in realism mode
            if self.realism_mode:
                soil_moisture_adj_mean = soil_moisture_mean * (humidity / humidity_mean)
                soil_moisture = round(self._generate_feature(soil_moisture_adj_mean, *soil_moisture_range, variation=0.10, dist="normal"))
            else:
                soil_moisture = round(self._generate_feature(soil_moisture_mean, *soil_moisture_range, variation=0.10, dist="normal"))
            
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
        """Ensure a crop reaches target sample size"""
        current_count = self.get_crop_count(crop_name)
        if current_count >= target_count:
            print(f"{crop_name} already has {current_count} samples (target: {target_count})")
            return
        
        needed = target_count - current_count
        self.add_crop_to_dataset(
            crop_name, needed, 
            ph_range, ec_range, humidity_range, 
            sunlight_range, soil_temp_range, soil_moisture_range,
            ph_mean, ec_mean, humidity_mean,
            sunlight_mean, soil_temp_mean, soil_moisture_mean
        )
    
    def get_crop_count(self, crop_name):
        """Count samples for crop"""
        if len(self.existing_data) == 0:
            return 0
        return len(self.existing_data[self.existing_data['label'] == crop_name])
    
    def save_dataset(self, output_path):
        """Save dataset to CSV"""
        # Only create directory if path contains directories
        dir_path = os.path.dirname(output_path)
        if dir_path:  # Only if there's actually a directory path
            os.makedirs(dir_path, exist_ok=True)
        self.existing_data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        # Update the path so future saves will use the same file
        self.existing_csv_path = output_path
    
    def show_dataset_summary(self):
        """Show summary of dataset"""
        if len(self.existing_data) == 0:
            print("Dataset is empty")
            return
        
        print(f"Total rows: {len(self.existing_data)}")
        print(f"Unique crops: {len(self.existing_data['label'].unique())}")
        print(self.existing_data['label'].value_counts())

# Example usage
if __name__ == "__main__":
    output_path = "enhanced_crop_data.csv"

    generator = CropDataGenerator(existing_csv_path=output_path, realism_mode=True)
    
    
 
   

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

    

    




 

    # Orange - subtropical, prefers drier air, cooler winters help fruiting
    generator.fill_crop_to_target(
        crop_name="Orange",
        target_count=100,
        ph_range=(5.5, 7.2), ph_mean=6.5,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(50, 75), humidity_mean=65,   # lower humidity tolerance
        sunlight_range=(40000, 85000), sunlight_mean=65000,
        soil_temp_range=(18, 28), soil_temp_mean=23,   
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

    

   

   

   

    
   
    
 

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")


 