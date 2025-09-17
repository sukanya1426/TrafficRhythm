import pandas as pd
import joblib
import numpy as np

def get_signal_timing(congestion_level, vehicle_ratio, base_time=20, max_time=40):
    """
    Map congestion level to green light duration, adjusted by vehicle ratio.
    
    Args:
        congestion_level (int): Congestion level (1=low, 2=medium, 3=high).
        vehicle_ratio (float): Ratio of vehicles in direction vs. total.
        base_time (int): Minimum green time (seconds).
        max_time (int): Maximum green time (seconds).
    
    Returns:
        int: Green light duration in seconds.
    """
    if congestion_level == 1:  # Low
        base = base_time
    elif congestion_level == 2:  # Medium
        base = (base_time + max_time) // 2
    else:  # High
        base = max_time
    # Adjust based on vehicle ratio (scale between 0.8 and 1.2)
    adjusted_time = int(base * min(max(0.8, vehicle_ratio), 1.2))
    return max(base_time, min(max_time, adjusted_time))

def map_timings(input_file='preprocessed_four_road_data.csv', 
                model_file='rf_model.pkl', 
                output_file='predicted_signal_timings.csv'):
    """
    Predict congestion levels and map to signal timings.
    
    Args:
        input_file (str): Path to the preprocessed CSV file.
        model_file (str): Path to the trained model.
        output_file (str): Path to save the output CSV with signal timings.
    """
    # Load preprocessed data
    df = pd.read_csv(input_file)
    
    # Load trained model
    model = joblib.load(model_file)
    
    # Prepare features for prediction
    X = df.drop(['congestion_level', 'time_of_day'], axis=1)
    
    # Predict congestion levels
    y_pred = model.predict(X)
    
    # Calculate vehicle ratios for North-South and East-West
    total_vehicles = (df['vehicle_count_north'] + df['vehicle_count_south'] + 
                     df['vehicle_count_east'] + df['vehicle_count_west'])
    ns_ratio = (df['vehicle_count_north'] + df['vehicle_count_south']) / total_vehicles.replace(0, 1)
    ew_ratio = (df['vehicle_count_east'] + df['vehicle_count_west']) / total_vehicles.replace(0, 1)
    
    # Map predictions to green light durations
    df['green_time_north_south'] = [get_signal_timing(y, r) for y, r in zip(y_pred, ns_ratio)]
    df['green_time_east_west'] = [get_signal_timing(y, r) for y, r in zip(y_pred, ew_ratio)]
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Signal timings saved to {output_file}")

if __name__ == "__main__":
    map_timings()