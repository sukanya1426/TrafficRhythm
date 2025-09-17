import pandas as pd
import numpy as np

def generate_traffic_data(n_days=7, output_file='four_road_traffic_data.csv'):
    """
    Generate a simulated traffic dataset for a four-road intersection with 5-minute intervals.
    
    Args:
        n_days (int): Number of days to simulate.
        output_file (str): Path to save the CSV file.
    """
    np.random.seed(42)
    
    # Generate 5-minute intervals for one day (24 hours * 12 intervals/hour = 288 intervals)
    minutes = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
    n_samples = len(minutes) * n_days
    
    # Simulate traffic patterns: higher vehicle counts during peak hours (7-10 AM, 4-7 PM)
    hours = np.array([int(t.split(':')[0]) for t in minutes] * n_days)
    peak_hours = (hours >= 7) & (hours < 10) | (hours >= 16) & (hours < 19)
    
    data = {
        'time_of_day': minutes * n_days,
        'day_of_week': np.repeat(np.arange(n_days), len(minutes)),
        'vehicle_count_north': np.random.randint(5, 30, n_samples) + peak_hours * np.random.randint(20, 40, n_samples),
        'vehicle_count_south': np.random.randint(5, 30, n_samples) + peak_hours * np.random.randint(20, 40, n_samples),
        'vehicle_count_east': np.random.randint(5, 30, n_samples) + peak_hours * np.random.randint(20, 40, n_samples),
        'vehicle_count_west': np.random.randint(5, 30, n_samples) + peak_hours * np.random.randint(20, 40, n_samples),
        'previous_signal_north_south': np.random.randint(20, 60, n_samples),
        'previous_signal_east_west': np.random.randint(20, 60, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate congestion level based on total vehicle count
    total_vehicles = (df['vehicle_count_north'] + df['vehicle_count_south'] + 
                     df['vehicle_count_east'] + df['vehicle_count_west'])
    df['congestion_level'] = pd.qcut(total_vehicles, q=3, labels=[1, 2, 3]).astype(int)  # 1=low, 2=medium, 3=high
    
    df.to_csv(output_file, index=False)
    print(f"Generated dataset with {n_samples} samples saved to {output_file}")

if __name__ == "__main__":
    generate_traffic_data()
