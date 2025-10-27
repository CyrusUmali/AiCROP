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
        crop_name="cacao",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.6,
        ec_range=(1000, 1600), ec_mean=1300,
        humidity_range=(75, 95), humidity_mean=85,
        sunlight_range=(15000, 40000), sunlight_mean=25000,  # lower due to shade needs
        soil_temp_range=(22, 28), soil_temp_mean=25,
        soil_moisture_range=(65, 85), soil_moisture_mean=75
    )

    # Guyabano - sun-tolerant, prefers well-drained soils
    generator.fill_crop_to_target(
        crop_name="guyabano",
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
        crop_name="lanzones",
        target_count=100,
        ph_range=(3.5, 6.0), ph_mean=4.8,
        ec_range=(850, 1100), ec_mean=950,
        humidity_range=(67, 80), humidity_mean=75,
        sunlight_range=(4300, 10000), sunlight_mean=8000,  # semi-shade
        soil_temp_range=(26, 31), soil_temp_mean=27,
        soil_moisture_range=(80, 99), soil_moisture_mean=90
    )

    # Durian - deep-rooted, high water demand, sun-loving
    generator.fill_crop_to_target(
        crop_name="durian",
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
        crop_name="rambutan",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(680, 780), ec_mean=720,
        humidity_range=(75, 85), humidity_mean=80,
        sunlight_range=(2000, 10000), sunlight_mean=5000,
        soil_temp_range=(25, 29), soil_temp_mean=27,
        soil_moisture_range=(85, 99), soil_moisture_mean=93
    )

    





    #  print("Citrus: calamansi, orange")  

      
    # === Citrus Group ===

    # Calamansi - tropical, tolerates humidity, grows well in lowland Philippines
    generator.fill_crop_to_target(
        crop_name="calamansi",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(65, 85), humidity_mean=75,   # slightly higher humidity than orange
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(23, 30), soil_temp_mean=27,  # warmer preference
        soil_moisture_range=(55, 75), soil_moisture_mean=65
    )

    # Orange - subtropical, prefers drier air, cooler winters help fruiting
    generator.fill_crop_to_target(
        crop_name="orange",
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

    # Snap Bean - prefers cooler, moist conditions
    generator.fill_crop_to_target(
        crop_name="snap bean",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 85), humidity_mean=72,
        sunlight_range=(35000, 70000), sunlight_mean=50000,  # not extreme sun
        soil_temp_range=(18, 26), soil_temp_mean=22,         # cooler soil preferred
        soil_moisture_range=(55, 80), soil_moisture_mean=68
    )

    # String Bean (Yardlong Bean) - tropical, heat-loving
    generator.fill_crop_to_target(
        crop_name="string bean",
        target_count=100,
        ph_range=(5.8, 6.8), ph_mean=6.3,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(55, 75), humidity_mean=65,           # less humidity sensitive
        sunlight_range=(50000, 90000), sunlight_mean=75000,  # loves strong sun
        soil_temp_range=(22, 32), soil_temp_mean=27,         # hotter than snap bean
        soil_moisture_range=(45, 70), soil_moisture_mean=55  # tolerates drier soil
    )

    # Sigarilyas (Winged Bean) - tropical, rainfall-adapted
    generator.fill_crop_to_target(
        crop_name="sigarilyas",
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
        crop_name="mungbean",
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

# Pechay - tropical, fast growth, high fertility & water demand
    generator.fill_crop_to_target(
        crop_name="pechay",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,                # higher fertility demand
        humidity_range=(70, 90), humidity_mean=82,          # thrives in humid tropics
        sunlight_range=(35000, 70000), sunlight_mean=50000, # moderate sun
        soil_temp_range=(20, 28), soil_temp_mean=24,        # warm tropical soil
        soil_moisture_range=(65, 85), soil_moisture_mean=75 # consistently moist soil
    )

    # Mustard - more cool-tolerant, lower fertility, tolerates drier soils
    generator.fill_crop_to_target(
        crop_name="mustard",
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

    # Ampalaya - tropical, heat-tolerant, moderate fertility
    generator.fill_crop_to_target(
        crop_name="ampalaya",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1200, 1800), ec_mean=1500,                 # moderate fertility
        humidity_range=(55, 75), humidity_mean=65,           # less humidity sensitive
        sunlight_range=(40000, 85000), sunlight_mean=60000,  # full sun
        soil_temp_range=(22, 32), soil_temp_mean=27,         # heat tolerant
        soil_moisture_range=(45, 70), soil_moisture_mean=55  # avoids waterlogging
    )

    # Patola (Sponge Gourd) - high fertility, water-loving
    generator.fill_crop_to_target(
        crop_name="patola",
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
        crop_name="upo",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1200, 2200), ec_mean=1700,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 80000), sunlight_mean=60000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(55, 80), soil_moisture_mean=65
    )

    # Squash (Calabaza) - drought-tolerant, wide pH and soil range
    generator.fill_crop_to_target(
        crop_name="squash",
        target_count=100,
        ph_range=(5.5, 7.2), ph_mean=6.4,                    # widest pH tolerance in group
        ec_range=(1000, 2000), ec_mean=1500,                 # moderate fertility
        humidity_range=(50, 75), humidity_mean=65,           # tolerates drier air
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 32), soil_temp_mean=26,
        soil_moisture_range=(40, 70), soil_moisture_mean=55  # drought-tolerant
    )


    
    


    # print("Peppers: sili panigang, sili tingala")

    
    # === Peppers Group ===

    # Sili Panigang (Long Green Chili) - vegetable use, prefers moderate humidity and moisture
    generator.fill_crop_to_target(
        crop_name="sili panigang",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(65, 85), humidity_mean=75,           # higher humidity preference
        sunlight_range=(40000, 80000), sunlight_mean=60000,  # moderate sun
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(55, 80), soil_moisture_mean=68  # needs steady soil moisture
    )

    # Sili Tingala (Bird’s Eye Chili) - hot pepper, stress-tolerant
    generator.fill_crop_to_target(
        crop_name="sili tingala",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1200, 2200), ec_mean=1700,                 # slightly less fertility need
        humidity_range=(50, 75), humidity_mean=62,           # tolerates lower humidity
        sunlight_range=(50000, 90000), sunlight_mean=75000,  # stronger sun preference
        soil_temp_range=(22, 34), soil_temp_mean=28,         # hotter tolerance
        soil_moisture_range=(40, 70), soil_moisture_mean=55  # tolerates drier soils
    )




    # print("Root Crops: sweet potato, ube, cassava")

    
    # === Root Crops Group ===

    # Sweet Potato - drought-tolerant, prefers sandy soils, fast grower
    generator.fill_crop_to_target(
        crop_name="sweet potato",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(800, 1800), ec_mean=1300,                  # moderate fertility
        humidity_range=(55, 75), humidity_mean=65,
        sunlight_range=(30000, 75000), sunlight_mean=50000,  # moderate to high sun
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(40, 70), soil_moisture_mean=55  # drought-tolerant
    )

    # Ube (Purple Yam) - needs fertile, moist soils for tuber development
    generator.fill_crop_to_target(
        crop_name="ube",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1000, 2000), ec_mean=1500,                 # higher fertility need
        humidity_range=(65, 85), humidity_mean=75,           # more humid than sweet potato
        sunlight_range=(25000, 65000), sunlight_mean=45000,  # less sun than cassava
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(55, 80), soil_moisture_mean=68  # higher water demand
    )

    # Cassava - extreme drought tolerance, can grow in poor soils
    generator.fill_crop_to_target(
        crop_name="cassava",
        target_count=100,
        ph_range=(5.0, 6.8), ph_mean=5.8,
        ec_range=(600, 1400), ec_mean=1000,                  # lowest fertility requirement
        humidity_range=(50, 75), humidity_mean=62,           # drier air tolerated
        sunlight_range=(40000, 90000), sunlight_mean=70000,  # loves full sun
        soil_temp_range=(24, 34), soil_temp_mean=28,         # hotter than others
        soil_moisture_range=(35, 65), soil_moisture_mean=50  # lowest moisture need
    )

 

    # print("Large Tropical Fruits: banana, jackfruit, coconut")

    # === Large Tropical Fruits Group ===

    # Banana - shallow roots, needs high fertility & moisture
    generator.fill_crop_to_target(
        crop_name="banana",
        target_count=100,
        ph_range=(4.0, 5.5), ph_mean=5.0,
        ec_range=(700, 1000), ec_mean=850,                 # very nutrient hungry
        humidity_range=(65, 85), humidity_mean=75,
        sunlight_range=(8000, 20000), sunlight_mean=15000,   
        soil_temp_range=(27, 32), soil_temp_mean=28,
        soil_moisture_range=(80, 99), soil_moisture_mean=90  # highest water demand
    )

    # Jackfruit - deep roots, less fertility and moisture demanding than banana
    generator.fill_crop_to_target(
        crop_name="jackfruit",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,                  # lower fertility need
        humidity_range=(65, 85), humidity_mean=75,
        sunlight_range=(30000, 75000), sunlight_mean=55000,  # tolerates partial sun
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(50, 75), soil_moisture_mean=62  # moderate water demand
    )

    # Coconut - hardy palm, tolerates salinity, thrives in coastal full sun
    generator.fill_crop_to_target(
        crop_name="coconut",
        target_count=100,
        ph_range=(5.0, 7.0), ph_mean=6.2,
        ec_range=(350, 500), ec_mean=420,
        humidity_range=(65, 90), humidity_mean=78,
        sunlight_range=(12000, 25000), sunlight_mean=17000,  # full sun, high light tolerance
        soil_temp_range=(24, 33), soil_temp_mean=29,
        soil_moisture_range=(80, 99), soil_moisture_mean=90  # more drought-tolerant than banana
    )


    


    # print("High-sun crops: maize, papaya, watermelon")

    # === High-sun Crops Group ===

    # Maize - heavy feeder, adaptable but needs consistent water during grain fill
    generator.fill_crop_to_target(
        crop_name="maize",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1200, 2200), ec_mean=1700,               # high nutrient demand
        humidity_range=(55, 75), humidity_mean=65,         # slightly lower tolerance
        sunlight_range=(55000, 100000), sunlight_mean=80000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    # Papaya - tropical, higher temperature requirement, shallow-rooted & sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="papaya",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1200, 2200), ec_mean=1700,
        humidity_range=(60, 85), humidity_mean=75,         # more humidity-tolerant than maize
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(25, 32), soil_temp_mean=28,       # prefers warmer soil
        soil_moisture_range=(55, 80), soil_moisture_mean=65
    )

    # Watermelon - drought-tolerant cucurbit, wide pH range, less fertility demand
    generator.fill_crop_to_target(
        crop_name="watermelon",
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
        crop_name="tomato",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(2000, 3000), ec_mean=2400,            # high nutrient demand
        humidity_range=(55, 75), humidity_mean=68,      # moderate humidity
        sunlight_range=(45000, 85000), sunlight_mean=65000,
        soil_temp_range=(16, 26), soil_temp_mean=22,    # cooler root temp preferred
        soil_moisture_range=(60, 80), soil_moisture_mean=70  # steady moisture for fruit set
    )

    # Eggplant - more heat-tolerant, slightly lower fertility requirement, tolerates wider conditions
    generator.fill_crop_to_target(
        crop_name="eggplant",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1500, 3000), ec_mean=2200,
        humidity_range=(60, 85), humidity_mean=72,      # tolerates higher humidity
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 32), soil_temp_mean=26,    # warmer root temps tolerated
        soil_moisture_range=(50, 75), soil_moisture_mean=62  # a bit more drought-tolerant
    )




    # print("Cool weather: onion, radish")


        # Onion - prefers cool to mild temps, moderate fertility, sensitive to waterlogging
    generator.fill_crop_to_target(
        crop_name="onion",
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
        crop_name="radish",
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
        crop_name="apple",
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
        crop_name="grapes",
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
        crop_name="rice",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(70, 95), humidity_mean=85,        # high humidity crop
        sunlight_range=(30000, 60000), sunlight_mean=45000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(80, 100), soil_moisture_mean=90  # flooded/paddy conditions
    )

    # Oyster Mushroom - shade crop, very high humidity, moderate temps
    generator.fill_crop_to_target(
        crop_name="oyster mushroom",
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
        crop_name="mango",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,
        ec_range=(800, 1600), ec_mean=1200,            # moderate fertility
        humidity_range=(45, 70), humidity_mean=55,     # not excessively humid for good flowering
        sunlight_range=(60000, 100000), sunlight_mean=80000,
        soil_temp_range=(24, 36), soil_temp_mean=30,   # warm root zone
        soil_moisture_range=(30, 60), soil_moisture_mean=45  # relatively drier than many tropical fruit trees
    )

    # Okra - heat-loving, full-sun annual; tolerates drier spells but performs with moderate fertility
    generator.fill_crop_to_target(
        crop_name="okra",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1200, 2200), ec_mean=1700,           # moderate-to-high fertility
        humidity_range=(60, 85), humidity_mean=72,
        sunlight_range=(60000, 100000), sunlight_mean=80000,
        soil_temp_range=(22, 34), soil_temp_mean=28,
        soil_moisture_range=(40, 75), soil_moisture_mean=55  # tolerates some dryness
    )

    # Pineapple - prefers acidic, well-drained soils; drought-tolerant relative, high light requirement
    generator.fill_crop_to_target(
        crop_name="pineapple",
        target_count=100,
        ph_range=(4.0, 6.0), ph_mean=5.0,
        ec_range=(430, 700), ec_mean=600,             # lower fertility
        humidity_range=(70, 76), humidity_mean=72,
        sunlight_range=(20000, 45000), sunlight_mean=36000,
        soil_temp_range=(31, 34), soil_temp_mean=32,
        soil_moisture_range=(95, 99), soil_moisture_mean=97# prefers well-drained, not waterlogged
    )

    # Gabi (Taro) - shade tolerant, high humidity & soil moisture, lower light
    generator.fill_crop_to_target(
        crop_name="gabi",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(75, 95), humidity_mean=85,     # very humid conditions
        sunlight_range=(10000, 40000), sunlight_mean=25000,  # shade to partial shade
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(70, 95), soil_moisture_mean=85  # likes saturated/very moist soils
    )

    # Ginger - shade/understory crop, high humidity and consistent moisture, moderate fertility
    generator.fill_crop_to_target(
        crop_name="ginger",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(70, 95), humidity_mean=82,     # prefers very humid microclimate
        sunlight_range=(5000, 35000), sunlight_mean=20000,   # low light / filtered shade
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(65, 90), soil_moisture_mean=78  # consistently moist but not waterlogged
    )

    
   
   


    

   

    


   
   

   

    

   

    
    

    

    

    

  

    
   

    

    
    

   

    
 
 

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")


 