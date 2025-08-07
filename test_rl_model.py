import numpy as np
import pandas as pd
from traffic_env import FourRoadTrafficEnv
import joblib

def test_rl_model(data_file='preprocessed_four_road_data.csv', 
                 q_table_file='q_table.pkl', 
                 output_file='rl_signal_timings.csv', 
                 n_cycles=100):
    """
    Test the RL model by simulating cycles and outputting green times.
    
    Args:
        data_file (str): Path to preprocessed traffic data CSV.
        q_table_file (str): Path to the trained Q-table.
        output_file (str): Path to save the output CSV with RL signal timings.
        n_cycles (int): Number of simulation cycles.
    """
    env = FourRoadTrafficEnv(data_file)
    q_table = joblib.load(q_table_file)
    
    results = []
    state = env.reset()
    
    for _ in range(n_cycles):
        state_idx = discretize_state(state)
        action = np.argmax(q_table[state_idx])
        green_ns, green_ew = env.actions[action]
        
        # Record results
        results.append({
            'time_of_day': env.data['time_of_day'].iloc[env.current_step],
            'vehicle_count_north': state[0],
            'vehicle_count_south': state[1],
            'vehicle_count_east': state[2],
            'vehicle_count_west': state[3],
            'green_time_north_south': green_ns,
            'green_time_east_west': green_ew
        })
        
        # Step environment
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"RL signal timings saved to {output_file}")

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

if __name__ == "__main__":
    test_rl_model()