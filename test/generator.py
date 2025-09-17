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
    
    # Add sweet potato data with new parameters
  

    generator.fill_crop_to_target(
        crop_name="ampalaya",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1200, 1800), ec_mean=1500,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(30000, 80000), sunlight_mean=55000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="apple",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(30000, 80000), sunlight_mean=55000,
        soil_temp_range=(12, 22), soil_temp_mean=17,
        soil_moisture_range=(50, 70), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="banana",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(25, 32), soil_temp_mean=28,
        soil_moisture_range=(60, 85), soil_moisture_mean=72
    )

    generator.fill_crop_to_target(
        crop_name="cacao",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(20000, 60000), sunlight_mean=40000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(60, 80), soil_moisture_mean=70
    )

    generator.fill_crop_to_target(
        crop_name="calamansi",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(50, 70), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="cassava",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(30000, 80000), sunlight_mean=55000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(40, 70), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="coconut",
        target_count=100,
        ph_range=(5.0, 7.0), ph_mean=6.0,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(24, 33), soil_temp_mean=28,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="durian",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(1200, 2000), ec_mean=1600,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(60, 80), soil_moisture_mean=70
    )

    generator.fill_crop_to_target(
        crop_name="eggplant",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1500, 3000), ec_mean=2200,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(35000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(55, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="gabi",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(65, 85), humidity_mean=75,
        sunlight_range=(20000, 50000), sunlight_mean=35000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(60, 80), soil_moisture_mean=70
    )

    generator.fill_crop_to_target(
        crop_name="ginger",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(60, 90), humidity_mean=75,
        sunlight_range=(20000, 60000), sunlight_mean=40000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(60, 85), soil_moisture_mean=72
    )

    generator.fill_crop_to_target(
        crop_name="grapes",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(50000, 100000), sunlight_mean=75000,
        soil_temp_range=(18, 28), soil_temp_mean=23,
        soil_moisture_range=(45, 65), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="guyabano",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(60, 80), soil_moisture_mean=70
    )

    generator.fill_crop_to_target(
        crop_name="jackfruit",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(65, 85), humidity_mean=75,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="lanzones",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="maize",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(50000, 100000), sunlight_mean=75000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="mango",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(40000, 100000), sunlight_mean=70000,
        soil_temp_range=(24, 35), soil_temp_mean=30,
        soil_moisture_range=(40, 60), soil_moisture_mean=50
    )

    generator.fill_crop_to_target(
        crop_name="mungbean",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(40, 70), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="mustard",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(30000, 80000), sunlight_mean=55000,
        soil_temp_range=(15, 25), soil_temp_mean=20,
        soil_moisture_range=(40, 70), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="okra",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(20, 32), soil_temp_mean=26,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="onion",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(40000, 80000), sunlight_mean=60000,
        soil_temp_range=(15, 25), soil_temp_mean=20,
        soil_moisture_range=(40, 70), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="orange",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(50, 70), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="oyster mushroom",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(500, 1500), ec_mean=1000,
        humidity_range=(80, 95), humidity_mean=88,
        sunlight_range=(5000, 20000), sunlight_mean=12000,
        soil_temp_range=(18, 25), soil_temp_mean=22,
        soil_moisture_range=(60, 85), soil_moisture_mean=72
    )

    generator.fill_crop_to_target(
        crop_name="papaya",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.3,
        ec_range=(1200, 2200), ec_mean=1700,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(25, 32), soil_temp_mean=28,
        soil_moisture_range=(50, 75), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="patola",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(55, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="pechay",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(18, 28), soil_temp_mean=23,
        soil_moisture_range=(60, 85), soil_moisture_mean=72
    )

    generator.fill_crop_to_target(
        crop_name="pineapple",
        target_count=100,
        ph_range=(4.5, 6.0), ph_mean=5.3,
        ec_range=(800, 1500), ec_mean=1100,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(50, 70), soil_moisture_mean=60
    )

    generator.fill_crop_to_target(
        crop_name="radish",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(50, 70), humidity_mean=60,
        sunlight_range=(40000, 80000), sunlight_mean=60000,
        soil_temp_range=(10, 20), soil_temp_mean=15,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="rambutan",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(1000, 1800), ec_mean=1400,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 70000), sunlight_mean=50000,
        soil_temp_range=(22, 30), soil_temp_mean=26,
        soil_moisture_range=(60, 85), soil_moisture_mean=70
    )

    generator.fill_crop_to_target(
        crop_name="rice",
        target_count=100,
        ph_range=(5.0, 6.5), ph_mean=5.8,
        ec_range=(800, 2000), ec_mean=1400,
        humidity_range=(70, 90), humidity_mean=80,
        sunlight_range=(30000, 60000), sunlight_mean=45000,
        soil_temp_range=(25, 35), soil_temp_mean=30,
        soil_moisture_range=(70, 95), soil_moisture_mean=85
    )

    generator.fill_crop_to_target(
        crop_name="sigarilyas",
        target_count=100,
        ph_range=(5.5, 6.8), ph_mean=6.2,
        ec_range=(1000, 2000), ec_mean=1500,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="sili panigang",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 85), humidity_mean=72,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 32), soil_temp_mean=26,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="sili tingala",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 85), humidity_mean=72,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 32), soil_temp_mean=26,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="snap bean",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(18, 30), soil_temp_mean=24,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="squash",
        target_count=100,
        ph_range=(5.5, 7.0), ph_mean=6.2,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=26,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="string bean",
        target_count=100,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        ec_range=(1500, 2500), ec_mean=2000,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(18, 30), soil_temp_mean=24,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="sweet potato",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(800, 1800), ec_mean=1300,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(25000, 70000), sunlight_mean=50000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(40, 70), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="tomato",
        target_count=100,
        ph_range=(5.8, 6.8), ph_mean=6.3,
        ec_range=(2000, 3500), ec_mean=2750,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(18, 30), soil_temp_mean=24,
        soil_moisture_range=(50, 75), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="ube",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(800, 1600), ec_mean=1200,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(25000, 70000), sunlight_mean=50000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(40, 70), soil_moisture_mean=55
    )

    generator.fill_crop_to_target(
        crop_name="upo",
        target_count=100,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        ec_range=(1200, 2200), ec_mean=1700,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(40000, 90000), sunlight_mean=65000,
        soil_temp_range=(20, 30), soil_temp_mean=25,
        soil_moisture_range=(55, 80), soil_moisture_mean=65
    )

    generator.fill_crop_to_target(
        crop_name="watermelon",
        target_count=100,
        ph_range=(5.5, 7.5), ph_mean=6.5,
        ec_range=(800, 1800), ec_mean=1300,
        humidity_range=(60, 80), humidity_mean=70,
        sunlight_range=(50000, 90000), sunlight_mean=70000,
        soil_temp_range=(22, 32), soil_temp_mean=27,
        soil_moisture_range=(50, 80), soil_moisture_mean=65
    )

 
 

    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")


 