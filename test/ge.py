import pandas as pd
import random
import numpy as np
import os

class CropDataGenerator:
    def __init__(self, existing_csv_path=None):
        """
        Initialize the crop data generator
        
        Parameters:
        existing_csv_path (str): Path to the existing CSV file (optional)
        """
        self.existing_csv_path = existing_csv_path
        self.existing_data = None
        
        if existing_csv_path and os.path.exists(existing_csv_path):
            self.load_existing_data()
        else:
            self.existing_data = pd.DataFrame(columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    
    def load_existing_data(self):
        """Load existing crop data"""
        try:
            self.existing_data = pd.read_csv(self.existing_csv_path)
            print(f"Loaded existing data: {len(self.existing_data)} rows")
            if len(self.existing_data) > 0:
                print(f"Existing crops: {list(self.existing_data['label'].unique())}")
                print(f"Number of unique crops: {len(self.existing_data['label'].unique())}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.existing_data = pd.DataFrame(columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    
    def generate_crop_data(self, crop_name, n_samples, n_range, p_range, k_range, 
                          temp_range, humidity_range, ph_range, rainfall_range,
                          n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                          humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """
        Generate synthetic data for a specific crop
        
        Parameters:
        crop_name (str): Name of the crop
        n_samples (int): Number of samples to generate
        n_range (tuple): (min, max) for Nitrogen
        p_range (tuple): (min, max) for Phosphorus
        k_range (tuple): (min, max) for Potassium
        temp_range (tuple): (min, max) for Temperature
        humidity_range (tuple): (min, max) for Humidity
        ph_range (tuple): (min, max) for pH
        rainfall_range (tuple): (min, max) for Rainfall
        n_mean (float): Mean/optimal value for Nitrogen (optional)
        p_mean (float): Mean/optimal value for Phosphorus (optional)
        k_mean (float): Mean/optimal value for Potassium (optional)
        temp_mean (float): Mean/optimal value for Temperature (optional)
        humidity_mean (float): Mean/optimal value for Humidity (optional)
        ph_mean (float): Mean/optimal value for pH (optional)
        rainfall_mean (float): Mean/optimal value for Rainfall (optional)
        
        Returns:
        list: List of rows for the crop
        """
        # Calculate means if not provided (midpoint of range)
        if n_mean is None:
            n_mean = (n_range[0] + n_range[1]) / 2
        if p_mean is None:
            p_mean = (p_range[0] + p_range[1]) / 2
        if k_mean is None:
            k_mean = (k_range[0] + k_range[1]) / 2
        if temp_mean is None:
            temp_mean = (temp_range[0] + temp_range[1]) / 2
        if humidity_mean is None:
            humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
        if ph_mean is None:
            ph_mean = (ph_range[0] + ph_range[1]) / 2
        if rainfall_mean is None:
            rainfall_mean = (rainfall_range[0] + rainfall_range[1]) / 2
        
        rows = []
        for _ in range(n_samples):
            # Generate values using normal distribution centered on mean
            # with standard deviation as 1/6 of the range (99.7% within range)
            
            # Nitrogen
            n_std = (n_range[1] - n_range[0]) / 6
            N = int(np.clip(np.random.normal(n_mean, n_std), n_range[0], n_range[1]))
            
            # Phosphorus
            p_std = (p_range[1] - p_range[0]) / 6
            P = int(np.clip(np.random.normal(p_mean, p_std), p_range[0], p_range[1]))
            
            # Potassium
            k_std = (k_range[1] - k_range[0]) / 6
            K = int(np.clip(np.random.normal(k_mean, k_std), k_range[0], k_range[1]))
            
            # Humidity
            humidity_std = (humidity_range[1] - humidity_range[0]) / 6
            humidity = int(np.clip(np.random.normal(humidity_mean, humidity_std), 
                                 humidity_range[0], humidity_range[1]))
            
            # Rainfall
            rainfall_std = (rainfall_range[1] - rainfall_range[0]) / 6
            rainfall = int(np.clip(np.random.normal(rainfall_mean, rainfall_std), 
                                 rainfall_range[0], rainfall_range[1]))
            
            # Temperature (float)
            temp_std = (temp_range[1] - temp_range[0]) / 6
            temperature = round(np.clip(np.random.normal(temp_mean, temp_std), 
                                      temp_range[0], temp_range[1]), 1)
            
            # pH (float)
            ph_std = (ph_range[1] - ph_range[0]) / 6
            ph = round(np.clip(np.random.normal(ph_mean, ph_std), 
                              ph_range[0], ph_range[1]), 2)
            
            rows.append([N, P, K, temperature, humidity, ph, rainfall, crop_name])
        
        return rows
    
    def add_crop_to_dataset(self, crop_name, n_samples, n_range, p_range, k_range, 
                           temp_range, humidity_range, ph_range, rainfall_range,
                           n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                           humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """
        Add a new crop with generated data to the dataset
        
        Parameters: Same as generate_crop_data
        """
        print(f"Generating {n_samples} samples for {crop_name}...")
        
        # Show means being used
        means = {
            'N': n_mean or (n_range[0] + n_range[1]) / 2,
            'P': p_mean or (p_range[0] + p_range[1]) / 2,
            'K': k_mean or (k_range[0] + k_range[1]) / 2,
            'Temperature': temp_mean or (temp_range[0] + temp_range[1]) / 2,
            'Humidity': humidity_mean or (humidity_range[0] + humidity_range[1]) / 2,
            'pH': ph_mean or (ph_range[0] + ph_range[1]) / 2,
            'Rainfall': rainfall_mean or (rainfall_range[0] + rainfall_range[1]) / 2
        }
        print(f"Using means: {means}")
        
        # Generate synthetic data
        synthetic_rows = self.generate_crop_data(
            crop_name, n_samples, n_range, p_range, k_range, 
            temp_range, humidity_range, ph_range, rainfall_range,
            n_mean, p_mean, k_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean
        )
        
        # Create DataFrame
        df_synthetic = pd.DataFrame(
            synthetic_rows, 
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
        )
        
        # Add to existing data
        self.existing_data = pd.concat([self.existing_data, df_synthetic], ignore_index=True)
        
        print(f"Added {n_samples} rows for {crop_name}")
        print(f"Total dataset size: {len(self.existing_data)} rows")
    
    def get_crop_count(self, crop_name):
        """Get current count of samples for a specific crop"""
        if len(self.existing_data) == 0:
            return 0
        return len(self.existing_data[self.existing_data['label'] == crop_name])
    
    def fill_crop_to_target(self, crop_name, target_count, n_range, p_range, k_range, 
                           temp_range, humidity_range, ph_range, rainfall_range,
                           n_mean=None, p_mean=None, k_mean=None, temp_mean=None,
                           humidity_mean=None, ph_mean=None, rainfall_mean=None):
        """
        Fill a crop to reach target number of samples
        
        Parameters:
        target_count (int): Target number of samples for the crop
        Other parameters: Same as generate_crop_data
        """
        current_count = self.get_crop_count(crop_name)
        
        if current_count >= target_count:
            print(f"{crop_name} already has {current_count} samples (target: {target_count})")
            return
        
        needed_samples = target_count - current_count
        self.add_crop_to_dataset(
            crop_name, needed_samples, n_range, p_range, k_range, 
            temp_range, humidity_range, ph_range, rainfall_range,
            n_mean, p_mean, k_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean
        )
    
    def save_dataset(self, output_path):
        """Save the complete dataset to CSV"""
        self.existing_data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Final dataset: {len(self.existing_data)} rows, {len(self.existing_data['label'].unique())} unique crops")
    
    def show_dataset_summary(self):
        """Show summary of current dataset"""
        if len(self.existing_data) == 0:
            print("Dataset is empty")
            return
            
        print("\nDataset Summary:")
        print(f"Total rows: {len(self.existing_data)}")
        print(f"Unique crops: {len(self.existing_data['label'].unique())}")
        print("\nCrop distribution:")
        crop_counts = self.existing_data['label'].value_counts()
        for crop, count in crop_counts.items():
            print(f"  {crop}: {count} samples")

# Example usage function
def example_usage():
    """Example of how to use the CropDataGenerator"""
    
    # Initialize generator (with or without existing data)
    generator = CropDataGenerator("existing_crop_data.csv")  # or None if no existing data
    
    # Add new crops with their parameter ranges and optimal means
    
    # Example 1: Add banana with specific means
    # generator.add_crop_to_dataset(
    #     crop_name="banana",
    #     n_samples=100,
    #     n_range=(80, 150), n_mean=115,      # Optimal N = 115
    #     p_range=(30, 50), p_mean=40,        # Optimal P = 40
    #     k_range=(40, 60), k_mean=50,        # Optimal K = 50
    #     temp_range=(22, 30), temp_mean=26,  # Optimal temp = 26°C
    #     humidity_range=(70, 90), humidity_mean=80,  # Optimal humidity = 80%
    #     ph_range=(5.5, 7.5), ph_mean=6.5,  # Optimal pH = 6.5
    #     rainfall_range=(150, 300), rainfall_mean=225  # Optimal rainfall = 225mm
    # )
    
    # Example 2: Add mango (means will be calculated as midpoint if not provided)
    # generator.add_crop_to_dataset(
    #     crop_name="mango",
    #     n_samples=100,
    #     n_range=(60, 120),
    #     p_range=(25, 45),
    #     k_range=(50, 90),
    #     temp_range=(24, 32),
    #     humidity_range=(60, 80),
    #     ph_range=(6.0, 7.0),
    #     rainfall_range=(200, 400)
    # )
    
    # Example 3: Fill existing crop to target count with specific optimal values
    generator.fill_crop_to_target(
        crop_name="rice",
        target_count=100,
        n_range=(60, 99), n_mean=79.89,      # High N requirement for rice
        p_range=(35, 60), p_mean=47.58,
        k_range=(35, 45), k_mean=39.87,
        temp_range=(20.05, 26.93), temp_mean=23.69,  # Warm climate optimal
        humidity_range=(80.12, 84.97), humidity_mean=82.27,  # High humidity for rice
        ph_range=(5.01, 7.87), ph_mean=6.43,  # Slightly acidic
        rainfall_range=(182.56, 298.56), rainfall_mean=236.18  # High water requirement
    )

    
    # Show summary
    generator.show_dataset_summary()
    
    # Save the final dataset
    generator.save_dataset("enhanced_crop_data.csv")

# Quick function for single crop generation (like your example)
def quick_generate_crop(crop_name, n_samples, existing_csv=None, output_csv="local_crop_data.csv"):
    """
    Quick function to generate data for one crop and append to existing CSV
    Similar to your original example
    """
    # Default parameter ranges (you can modify these)
    default_ranges = {
        'N': (20, 200),
        'P': (10, 100), 
        'K': (10, 200),
        'temperature': (10, 40),
        'humidity': (20, 100),
        'ph': (4.0, 9.0),
        'rainfall': (50, 2000)
    }
    
    print(f"Generating {n_samples} samples for {crop_name} with default ranges")
    print("Modify the default_ranges in the function if needed")
    
    generator = CropDataGenerator(existing_csv)
    generator.add_crop_to_dataset(
        crop_name, n_samples,
        default_ranges['N'], default_ranges['P'], default_ranges['K'],
        default_ranges['temperature'], default_ranges['humidity'], 
        default_ranges['ph'], default_ranges['rainfall']
    )
    generator.save_dataset(output_csv)

if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Or use quick function
    # quick_generate_crop("papaya", 50)