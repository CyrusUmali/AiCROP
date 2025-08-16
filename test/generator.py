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
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        )
        
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing crop data"""
        try:
            existing = pd.read_csv(self.existing_csv_path)
            if not existing.empty:
                # Ensure columns match
                expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
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
    
    def generate_crop_data(self, crop_name, n_samples, n_range, p_range, k_range, 
                           temp_range, humidity_range, ph_range, rainfall_range,
                           n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                           humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """
        Generate synthetic crop data with optional realism adjustments.
        """
        if n_mean is None: n_mean = (n_range[0] + n_range[1]) / 2
        if p_mean is None: p_mean = (p_range[0] + p_range[1]) / 2
        if k_mean is None: k_mean = (k_range[0] + k_range[1]) / 2
        if temp_mean is None: temp_mean = (temp_range[0] + temp_range[1]) / 2
        if humidity_mean is None: humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if ph_mean is None: ph_mean = (ph_range[0] + ph_range[1]) / 2
        if rainfall_mean is None: rainfall_mean = (rainfall_range[0] + rainfall_range[1]) / 2
        
        rows = []
        for _ in range(n_samples):
            # Nutrients: Slight skew to reflect real fertilization variability
            N = round(self._generate_feature(n_mean, *n_range, variation=0.12, dist="normal"))
            P = round(self._generate_feature(p_mean, *p_range, variation=0.15, dist="triangular"))
            K = round(self._generate_feature(k_mean, *k_range, variation=0.15, dist="triangular"))
            
            # Climate correlations
            temperature = round(self._generate_feature(temp_mean, *temp_range, variation=0.08, dist="normal"), 1)
            humidity = round(self._generate_feature(humidity_mean, *humidity_range, variation=0.05, dist="normal"), 1)
            
            # pH: narrow variation
            ph = round(self._generate_feature(ph_mean, *ph_range, variation=0.05, dist="normal"), 2)
            
            # Rainfall: correlated with humidity in realism mode
            if self.realism_mode:
                rainfall_adj_mean = rainfall_mean * (humidity / humidity_mean)
                rainfall = round(self._generate_feature(rainfall_adj_mean, *rainfall_range, variation=0.12, dist="normal"))
            else:
                rainfall = round(self._generate_feature(rainfall_mean, *rainfall_range, variation=0.12, dist="normal"))
            
            rows.append([N, P, K, temperature, humidity, ph, rainfall, crop_name])
        
        return rows
    
    def add_crop_to_dataset(self, crop_name, n_samples, n_range, p_range, k_range, 
                            temp_range, humidity_range, ph_range, rainfall_range,
                            n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                            humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """Add new crop data to dataset"""
        synthetic_rows = self.generate_crop_data(
            crop_name, n_samples, n_range, p_range, k_range,
            temp_range, humidity_range, ph_range, rainfall_range,
            n_mean, p_mean, k_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean
        )
        
        df_synthetic = pd.DataFrame(
            synthetic_rows, 
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
        )

        # Combine with existing data
        self.existing_data = pd.concat([self.existing_data, df_synthetic], ignore_index=True)
    
    def fill_crop_to_target(self, crop_name, target_count, n_range, p_range, k_range, 
                            temp_range, humidity_range, ph_range, rainfall_range,
                            n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                            humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """Ensure a crop reaches target sample size"""
        current_count = self.get_crop_count(crop_name)
        if current_count >= target_count:
            print(f"{crop_name} already has {current_count} samples (target: {target_count})")
            return
        
        needed = target_count - current_count
        self.add_crop_to_dataset(
            crop_name, needed, n_range, p_range, k_range,
            temp_range, humidity_range, ph_range, rainfall_range,
            n_mean, p_mean, k_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean
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

# Example
if __name__ == "__main__":
    output_path = "enhanced_crop_data.csv"

    generator = CropDataGenerator(existing_csv_path=output_path, realism_mode=True)
    

    generator.fill_crop_to_target(
        crop_name="ampalaya",
        target_count=100,
        n_range=(50, 100), n_mean=75,            # moderate N for vine crops
        p_range=(20, 40), p_mean=30,
        k_range=(40, 80), k_mean=60,
        temp_range=(25, 30), temp_mean=27,
        humidity_range=(60, 85), humidity_mean=72,
        ph_range=(5.5, 6.7), ph_mean=6.1,
        rainfall_range=(800, 1200), rainfall_mean=1000  # mm/year estimate
    )

    generator.fill_crop_to_target(
        crop_name="cacao",
        target_count=100,
        n_range=(100, 200), n_mean=150,
        p_range=(50, 200), p_mean=125,
        k_range=(100, 300), k_mean=200,
        temp_range=(21, 32), temp_mean=26.5,  # typical tropical range
        humidity_range=(80, 95), humidity_mean=87,
        ph_range=(5.0, 7.5), ph_mean=6.25,
        rainfall_range=(1500, 2000), rainfall_mean=1750
    )

    generator.fill_crop_to_target(
        crop_name="calamansi",
        target_count=100,
        n_range=(80, 150), n_mean=115,
        p_range=(30, 60), p_mean=45,
        k_range=(80, 150), k_mean=115,
        temp_range=(20, 32), temp_mean=26,
        humidity_range=(60, 90), humidity_mean=75,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(1000, 2000), rainfall_mean=1500
    )
    generator.fill_crop_to_target(
        crop_name="cassava",
        target_count=100,
        n_range=(50, 100), n_mean=75,
        p_range=(20, 40), p_mean=30,
        k_range=(50, 100), k_mean=75,
        temp_range=(18, 30), temp_mean=24,
        humidity_range=(50, 75), humidity_mean=62,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(1000, 1500), rainfall_mean=1250
    )


    generator.fill_crop_to_target(
        crop_name="cucumber",
        target_count=100,
        n_range=(80, 120), n_mean=100,
        p_range=(30, 50), p_mean=40,
        k_range=(80, 120), k_mean=100,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(60, 90), humidity_mean=75,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(600, 1000), rainfall_mean=800
    )


     



    generator.fill_crop_to_target(
        crop_name="eggplant",
        target_count=100,
        n_range=(80, 120), n_mean=100,
        p_range=(30, 60), p_mean=45,
        k_range=(80, 120), k_mean=100,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(60, 85), humidity_mean=72,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(600, 1200), rainfall_mean=900
    )

    generator.fill_crop_to_target(
        crop_name="durian",
        target_count=100,
        n_range=(150, 250), n_mean=200,
        p_range=(50, 100), p_mean=75,
        k_range=(200, 300), k_mean=250,
        temp_range=(22, 32), temp_mean=27,
        humidity_range=(70, 95), humidity_mean=82,
        ph_range=(5.0, 6.5), ph_mean=5.75,
        rainfall_range=(1800, 3000), rainfall_mean=2400
    )


    generator.fill_crop_to_target(
        crop_name="gabi",
        target_count=100,
        n_range=(80, 120), n_mean=100,
        p_range=(30, 60), p_mean=45,
        k_range=(80, 120), k_mean=100,
        temp_range=(18, 28), temp_mean=23,
        humidity_range=(70, 95), humidity_mean=82,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(1200, 2000), rainfall_mean=1600
    )


    generator.fill_crop_to_target(
        crop_name="ginger",
        target_count=100,
        n_range=(80, 100), n_mean=90,
        p_range=(30, 60), p_mean=45,
        k_range=(40, 90), k_mean=65,
        temp_range=(24, 29), temp_mean=26.5,
        humidity_range=(70, 90), humidity_mean=80,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(1500, 2500), rainfall_mean=2000
    )

    generator.fill_crop_to_target(
        crop_name="guyabano",
        target_count=100,
        n_range=(80, 150), n_mean=115,
        p_range=(30, 80), p_mean=55,
        k_range=(80, 150), k_mean=115,
        temp_range=(20, 32), temp_mean=26,
        humidity_range=(75, 95), humidity_mean=85,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(1200, 2500), rainfall_mean=1850
    )


    generator.fill_crop_to_target(
        crop_name="jackfruit",
        target_count=100,
        n_range=(100, 200), n_mean=150,
        p_range=(30, 60), p_mean=45,
        k_range=(100, 200), k_mean=150,
        temp_range=(24, 32), temp_mean=28,
        humidity_range=(60, 90), humidity_mean=75,
        ph_range=(6.0, 7.5), ph_mean=6.75,
        rainfall_range=(1500, 2500), rainfall_mean=2000
    )

    generator.fill_crop_to_target(
        crop_name="lanzones",
        target_count=100,
        n_range=(100, 200), n_mean=150,
        p_range=(30, 60), p_mean=45,
        k_range=(100, 200), k_mean=150,
        temp_range=(22, 35), temp_mean=28,
        humidity_range=(75, 95), humidity_mean=85,
        ph_range=(5.0, 6.5), ph_mean=5.75,
        rainfall_range=(1500, 3000), rainfall_mean=2200
    )

    

    generator.fill_crop_to_target(
        crop_name="mungbean",
        target_count=100,
        n_range=(34, 43), n_mean=38.5,
        p_range=(17.6, 21.7), p_mean=19.65,
        k_range=(53.2, 67.3), k_mean=60.25,
        temp_range=(25, 35), temp_mean=30,
        humidity_range=(70, 90), humidity_mean=80,
        ph_range=(6.2, 7.2), ph_mean=6.7,
        rainfall_range=(400, 550), rainfall_mean=475
    )


    generator.fill_crop_to_target(
        crop_name="mustard",
        target_count=100,
        n_range=(80, 120), n_mean=100,
        p_range=(25, 35), p_mean=30,
        k_range=(10, 20), k_mean=15,
        temp_range=(10, 25), temp_mean=18,
        humidity_range=(50, 80), humidity_mean=65,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(300, 600), rainfall_mean=450
    )


    generator.fill_crop_to_target(
        crop_name="onion",
        target_count=100,
        n_range=(95, 200), n_mean=147.5,
        p_range=(50, 300), p_mean=175,
        k_range=(70, 500), k_mean=285,
        temp_range=(10, 25), temp_mean=17.5,
        humidity_range=(50, 80), humidity_mean=65,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(400, 800), rainfall_mean=600
    )


    generator.fill_crop_to_target(
        crop_name="okra",
        target_count=100,
        n_range=(80, 150), n_mean=115,
        p_range=(50, 100), p_mean=75,
        k_range=(160, 300), k_mean=230,   
        temp_range=(25, 35), temp_mean=30,   
        humidity_range=(60, 90), humidity_mean=75,
        ph_range=(5.8, 6.8), ph_mean=6.3,
        rainfall_range=(400, 800), rainfall_mean=600
    )

    generator.fill_crop_to_target(
        crop_name="sili tingala",
        target_count=100,
        n_range=(100, 200), n_mean=150,
        p_range=(50, 100), p_mean=75,
        k_range=(100, 200), k_mean=150,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(70, 90), humidity_mean=80,
        ph_range=(5.5, 6.8), ph_mean=6.15,
        rainfall_range=(800, 1200), rainfall_mean=1000
    )

    generator.fill_crop_to_target(
        crop_name="squash",
        target_count=100,
        n_range=(50, 80), n_mean=65,
        p_range=(20, 40), p_mean=30,
        k_range=(60, 100), k_mean=80,
        temp_range=(18, 30), temp_mean=24,
        humidity_range=(60, 80), humidity_mean=70,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        rainfall_range=(500, 1000), rainfall_mean=750
    )

    generator.fill_crop_to_target(
        crop_name="pineapple",
        target_count=100,
        n_range=(250, 500), n_mean=375,
        p_range=(0, 50), p_mean=25,
        k_range=(350, 700), k_mean=525,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(50, 70), humidity_mean=60,
        ph_range=(4.5, 6.5), ph_mean=5.5,
        rainfall_range=(800, 2000), rainfall_mean=1400
    )

    generator.fill_crop_to_target(
        crop_name="radish",
        target_count=100,
        n_range=(2.15, 2.47), n_mean=2.31,
        p_range=(0.45, 0.51), p_mean=0.48,
        k_range=(2.58, 2.96), k_mean=2.77,
        temp_range=(10, 21), temp_mean=15,
        humidity_range=(50, 90), humidity_mean=70,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        rainfall_range=(300, 700), rainfall_mean=500
    )

    generator.fill_crop_to_target(
        crop_name="rambutan",
        target_count=100,
        n_range=(100, 300), n_mean=200,
        p_range=(115, 575), p_mean=345,
        k_range=(55, 275), k_mean=165,
        temp_range=(22, 34), temp_mean=28,
        humidity_range=(75, 80), humidity_mean=77.5,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(2000, 3000), rainfall_mean=2500
    )


    generator.fill_crop_to_target(
        crop_name="oyster mushroom",
        target_count=100,
        n_range=(0.8, 1.2), n_mean=1.0,    
        p_range=(0.4, 0.6), p_mean=0.5,  
        k_range=(0.4, 0.6), k_mean=0.5,    
        temp_range=(20, 30), temp_mean=25, humidity_range=(85, 95), humidity_mean=90,
    ph_range=(6.0, 7.5), ph_mean=6.75, rainfall_range=(0, 0), rainfall_mean=0        
    )

    generator.fill_crop_to_target(
        crop_name="sili panigang",
        target_count=100,
        n_range=(100, 150), n_mean=125,
        p_range=(50, 80), p_mean=65,
        k_range=(150, 200), k_mean=175,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(60, 90), humidity_mean=75,
        ph_range=(5.5, 6.8), ph_mean=6.15,
        rainfall_range=(800, 1200), rainfall_mean=1000
    )
    

 

    generator.fill_crop_to_target(
        crop_name="sigarilyas",
        target_count=100,
        n_range=(100, 200), n_mean=150,
        p_range=(40, 80), p_mean=60,
        k_range=(80, 160), k_mean=120,
        temp_range=(24, 32), temp_mean=28,
        humidity_range=(70, 90), humidity_mean=80,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(1200, 1500), rainfall_mean=1350
    )


    generator.fill_crop_to_target(
        crop_name="patola",
        target_count=100,
        n_range=(70, 120), n_mean=95,
        p_range=(40, 80), p_mean=60,
        k_range=(60, 100), k_mean=80,
        temp_range=(24, 32), temp_mean=28,
        humidity_range=(65, 90), humidity_mean=77,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(800, 1500), rainfall_mean=1150
    )

    generator.fill_crop_to_target(
        crop_name="pechay",
        target_count=100,
        n_range=(100, 150), n_mean=125,
        p_range=(50, 100), p_mean=75,
        k_range=(80, 150), k_mean=115,
        temp_range=(18, 22), temp_mean=20,
        humidity_range=(60, 85), humidity_mean=72.5,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(600, 1200), rainfall_mean=900
    )


    


  

    generator.fill_crop_to_target(
        crop_name="string bean",
        target_count=100,
        n_range=(50, 100), n_mean=75,
        p_range=(30, 60), p_mean=45,
        k_range=(50, 100), k_mean=75,
        temp_range=(18, 30), temp_mean=24,
        humidity_range=(60, 80), humidity_mean=70,
        ph_range=(5.8, 6.8), ph_mean=6.3,
        rainfall_range=(400, 800), rainfall_mean=600
    )

    generator.fill_crop_to_target(
        crop_name="sweet potato",
        target_count=100,
        n_range=(55, 120), n_mean=87.5,
        p_range=(60, 90), p_mean=75,
        k_range=(150, 300), k_mean=225,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(60, 80), humidity_mean=70,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(600, 1200), rainfall_mean=900
    )

    generator.fill_crop_to_target(
        crop_name="tomato",
        target_count=100,
        n_range=(150, 300), n_mean=225,
        p_range=(85, 200), p_mean=142.5,
        k_range=(200, 300), k_mean=250,
        temp_range=(18, 30), temp_mean=24,
        humidity_range=(50, 80), humidity_mean=65,
        ph_range=(5.5, 7.0), ph_mean=6.25,
        rainfall_range=(600, 1000), rainfall_mean=800
    )

    generator.fill_crop_to_target(
        crop_name="ube",
        target_count=100,
        n_range=(80, 150), n_mean=115,
        p_range=(40, 90), p_mean=65,
        k_range=(150, 300), k_mean=225,
        temp_range=(21, 30), temp_mean=25,
        humidity_range=(60, 80), humidity_mean=70,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(1000, 2000), rainfall_mean=1500
    )

    generator.fill_crop_to_target(
        crop_name="upo",
        target_count=100,
        n_range=(104, 164), n_mean=134,
        p_range=(54, 114), p_mean=84,
        k_range=(104, 164), k_mean=134,
        temp_range=(20, 30), temp_mean=25,
        humidity_range=(60, 80), humidity_mean=70,
        ph_range=(6.0, 7.0), ph_mean=6.5,
        rainfall_range=(600, 1200), rainfall_mean=900
    )

    generator.fill_crop_to_target(
        crop_name="snap bean",
        target_count=100,
        n_range=(70, 120), n_mean=90,
        p_range=(60, 110), p_mean=80,
        k_range=(80, 130), k_mean=100,
        temp_range=(22, 32), temp_mean=27,
        humidity_range=(70, 90), humidity_mean=80,
        ph_range=(5.5, 6.5), ph_mean=6.0,
        rainfall_range=(1500, 2500), rainfall_mean=2000
    )




    

    


    generator.show_dataset_summary()
    generator.save_dataset("enhanced_crop_data.csv")