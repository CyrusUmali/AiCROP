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
    
    

    # print("Tropical Fruits (High humidity, warm): cacao, guyabano, lanzones, durian, rambutan")

    
    # === Tropical Fruits (High humidity, warm) ===

    # Cacao - understory crop, shade-tolerant, very sensitive to drought
    generator.fill_crop_to_target(
        crop_name="Cacao",
        target_count=100,
        ph_range=(5.4, 6.5), ph_mean=6,
        ec_range=(420, 460), ec_mean=440,
        humidity_range=(70, 88), humidity_mean=85,
        sunlight_range=(800, 1100), sunlight_mean=900,  # lower due to shade needs
        soil_temp_range=(25, 28), soil_temp_mean=26.9,
        soil_moisture_range=(65, 85), soil_moisture_mean=75
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

    # Lanzones - very sensitive to drought, thrives in shaded/humid conditions
    generator.fill_crop_to_target(
        crop_name="Lanzones",
        target_count=100,
        ph_range=(4, 6.0), ph_mean=5.3,
        ec_range=(488, 950), ec_mean=750,
        humidity_range=(66, 86), humidity_mean=78,
        sunlight_range=(800, 4500), sunlight_mean=2700,   
        soil_temp_range=(26, 31), soil_temp_mean=27,
        soil_moisture_range=(83, 99), soil_moisture_mean=90
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
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(488, 760), ec_mean=600,
        humidity_range=(80, 87), humidity_mean=84   ,
        sunlight_range=(800, 2300), sunlight_mean=1500,
        soil_temp_range=(26, 28.5), soil_temp_mean=27,
        soil_moisture_range=(85, 99), soil_moisture_mean=93
    )

    





    #  print("Citrus: calamansi, orange")  

      
    # === Citrus Group ===

    # Calamansi - tropical, tolerates humidity, grows well in lowland Philippines
    generator.fill_crop_to_target(
        crop_name="Calamansi",
        target_count=100,
        ph_range=(4.8, 5.3), ph_mean=5,
        ec_range=(430, 442), ec_mean=435,
        humidity_range=(77, 81), humidity_mean=79,   
        sunlight_range=(1800, 2100), sunlight_mean=2000,
        soil_temp_range=(27.8, 29.3), soil_temp_mean=28.1,   
        soil_moisture_range=(78, 86), soil_moisture_mean=82
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






    print("Legumes: snap bean, string bean, sigarilyas, mungbean")

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

    # (a)String Bean (Yardlong Bean) - tropical, heat-loving
    generator.fill_crop_to_target(
        crop_name="String Bean",
        target_count=100,
        ph_range=(6.3, 7.5), ph_mean=7,
        ec_range=(500, 700), ec_mean=600,
        humidity_range=(68, 80), humidity_mean=76,         
        sunlight_range=(1000, 13000), sunlight_mean=7000,  
        soil_temp_range=(26, 32), soil_temp_mean=29,        
        soil_moisture_range=(85, 93), soil_moisture_mean=88   
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



    # print("Leafy Greens: pechay, mustard")


    
    # === Leafy Greens Group ===

#(*) Pechay - tropical, fast growth, high fertility & water demand
    generator.fill_crop_to_target(
        crop_name="Pechay",
        target_count=100,
        ph_range=(6.3, 7.5), ph_mean=6.8,
        ec_range=(450, 550), ec_mean=500,             
        humidity_range=(70, 85), humidity_mean=78,            
        sunlight_range=(1000, 3000), sunlight_mean=2000,
        soil_temp_range=(27, 29), soil_temp_mean=24,          
        soil_moisture_range=(92, 97), soil_moisture_mean=95  
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





    #  print("Gourds: ampalaya, patola, upo, squash")



    # === Gourds Group ===

    #(a) Ampalaya - tropical, heat-tolerant, moderate fertility
    generator.fill_crop_to_target(
        crop_name="Ampalaya",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(523, 532), ec_mean=527,                
        humidity_range=(72, 77), humidity_mean=74,          
        sunlight_range=(2000, 5000), sunlight_mean=3200,  
        soil_temp_range=(28, 31), soil_temp_mean=29.5,        
        soil_moisture_range=(87, 93), soil_moisture_mean=90   
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

    # Upo (Bottle Gourd) - adaptable, prefers moisture
    generator.fill_crop_to_target(
        crop_name="Upo",
        target_count=100,
        ph_range=(6.8, 7.2), ph_mean=7,
        ec_range=(530, 560), ec_mean=540,
        humidity_range=(70, 80), humidity_mean=76,
        sunlight_range=(2100, 3000), sunlight_mean=2700,
        soil_temp_range=(26.8, 28.5), soil_temp_mean=27.9,
        soil_moisture_range=(91, 94), soil_moisture_mean=93.5
    )

    #(L) Squash (Calabaza) - drought-tolerant, wide pH and soil range
    generator.fill_crop_to_target(
        crop_name="Squash",
        target_count=100,
        ph_range=(5.8, 7.2), ph_mean=6.5,                     
        ec_range=(500, 600), ec_mean=550,                  
        humidity_range=(68, 80), humidity_mean=75,           
        sunlight_range=(2000, 10000), sunlight_mean=6000,
        soil_temp_range=(27, 32), soil_temp_mean=30,
        soil_moisture_range=(86, 93), soil_moisture_mean=90  # drought-tolerant
    )




    generator.fill_crop_to_target(
        crop_name="Avocado",
        target_count=100,
        ph_range=(5.7, 6.3), ph_mean=6.0,
        ec_range=(453, 458), ec_mean=455,                
        humidity_range=(83, 88), humidity_mean=85,          
        sunlight_range=(1000, 4000), sunlight_mean=1150,  
        soil_temp_range=(26, 28), soil_temp_mean=27.1,        
        soil_moisture_range=(83, 88), soil_moisture_mean=85   
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
        crop_name="Cowpea",
        target_count=100,
        ph_range=(5.8, 6.3), ph_mean=6.0,
        ec_range=(406, 413), ec_mean=410,                
        humidity_range=(76, 80), humidity_mean=78,          
        sunlight_range=(2100, 2300), sunlight_mean=2200,  
        soil_temp_range=(27.3, 29), soil_temp_mean=28.4,        
        soil_moisture_range=(81, 86), soil_moisture_mean=83    
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
        crop_name="Bush Sitao",
        target_count=100,
        ph_range=(6.5, 7.2), ph_mean=7,
        ec_range=(530, 538), ec_mean=535,                
        humidity_range=(73, 78), humidity_mean=75,          
        sunlight_range=(2450, 2850), sunlight_mean=2600,  
        soil_temp_range=(26, 29), soil_temp_mean=28.7,        
        soil_moisture_range=(90, 98), soil_moisture_mean=94   
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
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(450, 550), ec_mean=500,                 
        humidity_range=(50, 75), humidity_mean=62,           
        sunlight_range=(2000, 9000), sunlight_mean=5000,  
        soil_temp_range=(27, 31), soil_temp_mean=29,         
        soil_moisture_range=(80, 89), soil_moisture_mean=85   
    )




    # print("Root Crops: sweet potato, ube, cassava")

    
    # === Root Crops Group ===

    #(a) Sweet Potato - drought-tolerant, prefers sandy soils, fast grower
    generator.fill_crop_to_target(
        crop_name="Sweet Potato",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(460, 600), ec_mean=530,                 
        humidity_range=(78, 86), humidity_mean=82 ,
        sunlight_range=(1000, 12000), sunlight_mean=6000,   
        soil_temp_range=(25 , 30), soil_temp_mean=27,
        soil_moisture_range=(88, 93), soil_moisture_mean=90  
    )

    generator.fill_crop_to_target(
        crop_name="Kamoteng Baging",
        target_count=100,
        ph_range=(4.9, 6.1), ph_mean=5,
        ec_range=(340, 368), ec_mean=359,                 
        humidity_range=(68, 74), humidity_mean=70 ,
        sunlight_range=(1000, 16000), sunlight_mean=10000,   
        soil_temp_range=(25 , 35), soil_temp_mean=34.6,
        soil_moisture_range=(88, 99), soil_moisture_mean=95
    )

    generator.fill_crop_to_target(
        crop_name="Katuray",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6,
        ec_range=(340, 368), ec_mean=354,                 
        humidity_range=(75, 81), humidity_mean=78 ,
        sunlight_range=(2000, 2500), sunlight_mean=2250,   
        soil_temp_range=(28.5 , 29), soil_temp_mean=28.4,
        soil_moisture_range=(80, 88), soil_moisture_mean=83
    )

    generator.fill_crop_to_target(
        crop_name="Kulo",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6,
        ec_range=(450, 480), ec_mean=470,                 
        humidity_range=(80, 88), humidity_mean=84 ,
        sunlight_range=(1030, 1080), sunlight_mean=1050,   
        soil_temp_range=(26.6 , 28), soil_temp_mean=27.4,
        soil_moisture_range=(80, 88), soil_moisture_mean=83
    )

    generator.fill_crop_to_target(
        crop_name="Lipute",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6,
        ec_range=(450, 480), ec_mean=420,                 
        humidity_range=(78, 84), humidity_mean=81 ,
        sunlight_range=(935, 965), sunlight_mean=950 ,   
        soil_temp_range=(26.6 , 28), soil_temp_mean=27.4,
        soil_moisture_range=(78, 83), soil_moisture_mean=80
    )

    generator.fill_crop_to_target(
        crop_name="Sweet Sorghum",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(460, 600), ec_mean=515,                 
        humidity_range=(75, 86), humidity_mean=80 ,
        sunlight_range=(1500, 12000), sunlight_mean=6000,   
        soil_temp_range=(25 , 30), soil_temp_mean=28.4,
        soil_moisture_range=(88, 93), soil_moisture_mean=89  
    )

    # Ube (Purple Yam) - needs fertile, moist soils for tuber development
    generator.fill_crop_to_target(
        crop_name="Ube",
        target_count=100,
        ph_range=(4.8, 5.4), ph_mean=5.0,
        ec_range=(466, 475), ec_mean=470,                 
        humidity_range=(81, 85), humidity_mean=83,           
        sunlight_range=(1200, 5000), sunlight_mean=3200,   
        soil_temp_range=(26, 28.2), soil_temp_mean=27.9,
        soil_moisture_range=(88, 92), soil_moisture_mean=90  
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

 

    # print("Large Tropical Fruits: banana, jackfruit, coconut")

    # === Large Tropical Fruits Group ===

    # Banana - shallow roots, needs high fertility & moisture
    generator.fill_crop_to_target(
        crop_name="Banana",
        target_count=100,
        ph_range=(5, 7.1), ph_mean=5.5,
        ec_range=(468, 556), ec_mean=500,                 # very nutrient hungry
        humidity_range=(73, 85), humidity_mean= 80,
        sunlight_range=(1280, 2520), sunlight_mean=1800,   
        soil_temp_range=(27, 30), soil_temp_mean=28.5,
        soil_moisture_range=(88, 97), soil_moisture_mean=93  
    )

    # Jackfruit - deep roots, less fertility and moisture demanding than banana
    generator.fill_crop_to_target(
        crop_name="Jackfruit",
        target_count=100,
        ph_range=(5.5, 6.3), ph_mean=6,
        ec_range=(465, 475), ec_mean=470,                  # lower fertility need
        humidity_range=(80, 87), humidity_mean=84,
        sunlight_range=(1020, 1080), sunlight_mean=1050,  # tolerates partial sun
        soil_temp_range=(26, 29), soil_temp_mean=27.4,
        soil_moisture_range=(87, 91), soil_moisture_mean=89  # moderate water demand
    )

    # Coconut - hardy palm, tolerates salinity, thrives in coastal full sun
    generator.fill_crop_to_target(
        crop_name="Coconut",
        target_count=100,
        ph_range=(5.8, 7.0), ph_mean=6.4,
        ec_range=(350, 520), ec_mean=440,
        humidity_range=(69, 86), humidity_mean=75,
        sunlight_range=(1200, 13000), sunlight_mean=6000,  
        soil_temp_range=(26.8, 34), soil_temp_mean=29,
        soil_moisture_range=(84, 99), soil_moisture_mean=90   
    )


    


    # print("High-sun crops: maize, papaya, watermelon")

    # === High-sun Crops Group ===

    # Maize - heavy feeder, adaptable but needs consistent water during grain fill
    generator.fill_crop_to_target(
        crop_name="Maize",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(477, 524), ec_mean=500,               # high nutrient demand
        humidity_range=(75, 85), humidity_mean=80,         # slightly lower tolerance
        sunlight_range=(1400, 3000), sunlight_mean=2200,
        soil_temp_range=(27, 30), soil_temp_mean=28.5,
        soil_moisture_range=(84, 92), soil_moisture_mean=65
    )

    # Papaya - tropical, higher temperature requirement, shallow-rooted & sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="Papaya",
        target_count=100,
        ph_range=(5.8, 6.3), ph_mean=6.0,
        ec_range=(428, 442), ec_mean=435,
        humidity_range=(76, 81), humidity_mean=79,         # more humidity-tolerant than maize
        sunlight_range=(1500, 8000), sunlight_mean=5000,
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


    



    # print("Nightshades: tomato, eggplant")

    
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

    #(a) Eggplant - more heat-tolerant, slightly lower fertility requirement, tolerates wider conditions
    generator.fill_crop_to_target(
        crop_name="Eggplant",
        target_count=100,
        ph_range=(5.8, 6.3), ph_mean=6,
        ec_range=(482, 488), ec_mean=485,
        humidity_range=(74, 78), humidity_mean=76,       
        sunlight_range=(2300, 2500), sunlight_mean=2400,
        soil_temp_range=(28.5, 29.3), soil_temp_mean=28.9,    
        soil_moisture_range=(85, 89), soil_moisture_mean=87   
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

    
    

    # print("Unique requirements: apple, grapes, rice, oyster mushroom, mango, okra, pineapple, gabi, ginger")


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





   
   
    # Mango - drought-tolerant tropical tree, full sun, warmer soils, lower soil moisture for good fruiting
    generator.fill_crop_to_target(
        crop_name="Mango",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,
        ec_range=(400, 530), ec_mean=460,           
        humidity_range=(78, 90), humidity_mean=85,    
        sunlight_range=(980, 1110), sunlight_mean=1050,
        soil_temp_range=(26, 30), soil_temp_mean=29.5,  
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

    # (L)Pineapple - prefers acidic, well-drained soils; drought-tolerant relative, high light requirement
    generator.fill_crop_to_target(
        crop_name="Pineapple",
        target_count=100,
        ph_range=(4.5, 6.0), ph_mean=5.3,
        ec_range=(430, 650), ec_mean=540,              
        humidity_range=(70, 78), humidity_mean=75,
        sunlight_range=(2500, 45000), sunlight_mean=20000,
        soil_temp_range=(28, 34.5), soil_temp_mean=31,
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


 