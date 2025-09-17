import pandas as pd
import numpy as np
import joblib
from traffic_env import FourRoadTrafficEnv

def discretize_state(state, bins=10):
    """
    Discretize vehicle counts to reduce state space.
    
    Args:
        state (list): [north, south, east, west, green_time].
        bins (int): Number of bins for vehicle counts.
    
    Returns:
        tuple: Discretized state indices.
    """
    vehicle_bins = np.linspace(-3, 3, bins)
    state_idx = [
        np.digitize(state[0], vehicle_bins),
        np.digitize(state[1], vehicle_bins),
        np.digitize(state[2], vehicle_bins),
        np.digitize(state[3], vehicle_bins)
    ]
    return tuple(np.clip(state_idx, 0, bins-1))

def integrate_models(data_file='preprocessed_four_road_data.csv', 
                     supervised_model_file='rf_model.pkl', 
                     q_table_file='q_table.pkl', 
                     output_file='four_road_output.csv'):
    """
    Integrate supervised and RL models to recommend signal timings.
    
    Args:
        data_file (str): Path to preprocessed traffic data CSV.
        supervised_model_file (str): Path to trained Random Forest model.
        q_table_file (str): Path to trained Q-table.
        output_file (str): Path to save integrated signal timings CSV.
    """
    # Load data and models
    df = pd.read_csv(data_file)
    supervised_model = joblib.load(supervised_model_file)
    q_table = joblib.load(q_table_file)
    env = FourRoadTrafficEnv(data_file)
    
    # Predict congestion levels using supervised model
    X = df.drop(['congestion_level', 'time_of_day'], axis=1)
    congestion_levels = supervised_model.predict(X)
    
    # Initialize results
    results = []
    
    # Process each time step
    for i in range(len(df)):
        # Get current state from dataset
        state = [
            df['vehicle_count_north'].iloc[i],
            df['vehicle_count_south'].iloc[i],
            df['vehicle_count_east'].iloc[i],
            df['vehicle_count_west'].iloc[i],
            20  # Default green time
        ]
        
        # Discretize state for Q-table
        state_idx = discretize_state(state)
        
        # Bias RL action selection with supervised congestion level
        congestion = congestion_levels[i]
        if congestion == 3:  # High congestion: prefer longer green times
            action_weights = [0.1, 0.3, 0.6, 0.1, 0.3, 0.6]
        elif congestion == 1:  # Low congestion: prefer shorter green times
            action_weights = [0.6, 0.3, 0.1, 0.6, 0.3, 0.1]
        else:  # Medium congestion: neutral
            action_weights = [1/6] * 6
        
        # Combine Q-table scores with weights
        q_values = q_table[state_idx]
        weighted_q_values = q_values * action_weights
        action = np.argmax(weighted_q_values)
        
        # Get green times
        green_ns, green_ew = env.actions[action]
        
        # Store results
        results.append({
            'time_of_day': df['time_of_day'].iloc[i],
            'vehicle_count_north': df['vehicle_count_north'].iloc[i],
            'vehicle_count_south': df['vehicle_count_south'].iloc[i],
            'vehicle_count_east': df['vehicle_count_east'].iloc[i],
            'vehicle_count_west': df['vehicle_count_west'].iloc[i],
            'congestion_level': congestion,
            'green_time_north_south': green_ns,
            'green_time_east_west': green_ew
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Integrated signal timings saved to {output_file}")

if __name__ == "__main__":
    integrate_models()
