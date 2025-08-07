import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

def calculate_waiting_time(df, ns_green_col, ew_green_col, vehicle_cols):
    """
    Calculate waiting time (vehicles on red lights) for each time step.
    
    Args:
        df (pd.DataFrame): DataFrame with vehicle counts and green times.
        ns_green_col (str): Column name for North-South green time.
        ew_green_col (str): Column name for East-West green time.
        vehicle_cols (list): List of vehicle count columns [north, south, east, west].
    
    Returns:
        np.array: Waiting times for each time step.
    """
    waiting_times = []
    for _, row in df.iterrows():
        if row[ns_green_col] > 0:  # North-South green, East-West red
            waiting_time = row[vehicle_cols[2]] + row[vehicle_cols[3]]  # East + West
        else:  # East-West green, North-South red
            waiting_time = row[vehicle_cols[0]] + row[vehicle_cols[1]]  # North + South
        waiting_times.append(max(0, waiting_time))  # Ensure non-negative
    return np.array(waiting_times)

def evaluate_congestion(ai_file='four_road_output.csv', 
                       baseline_file='four_road_traffic_data.csv', 
                       output_dir='plots'):
    """
    Compare congestion (waiting time) between AI-based and baseline signal timings.
    
    Args:
        ai_file (str): Path to integrated AI signal timings CSV.
        baseline_file (str): Path to original traffic data with baseline timings.
        output_dir (str): Directory to save the comparison plot.
    """
    try:
        # Load data
        ai_df = pd.read_csv(ai_file)
        baseline_df = pd.read_csv(baseline_file)
        
        # Ensure compatibility (same time steps)
        ai_df = ai_df.sort_values('time_of_day').reset_index(drop=True)
        baseline_df = baseline_df.sort_values('time_of_day').reset_index(drop=True)
        
        if len(ai_df) != len(baseline_df):
            raise ValueError("AI and baseline datasets have different lengths.")
        
        # Use non-normalized vehicle counts from baseline file
        vehicle_cols = ['vehicle_count_north', 'vehicle_count_south', 
                       'vehicle_count_east', 'vehicle_count_west']
        ai_df[vehicle_cols] = baseline_df[vehicle_cols]
        
        # Calculate waiting times
        ai_waiting_times = calculate_waiting_time(ai_df, 
                                                 'green_time_north_south', 
                                                 'green_time_east_west', 
                                                 vehicle_cols)
        baseline_waiting_times = calculate_waiting_time(baseline_df, 
                                                       'previous_signal_north_south', 
                                                       'previous_signal_east_west', 
                                                       vehicle_cols)
        
        # Compute average waiting times
        ai_avg_waiting = np.mean(ai_waiting_times)
        baseline_avg_waiting = np.mean(baseline_waiting_times)
        print(f"Average Waiting Time (AI): {ai_avg_waiting:.2f} vehicles")
        print(f"Average Waiting Time (Baseline): {baseline_avg_waiting:.2f} vehicles")
        
        # Calculate percentage reduction (avoid division by zero)
        if baseline_avg_waiting > 0:
            reduction = (baseline_avg_waiting - ai_avg_waiting) / baseline_avg_waiting * 100
        else:
            reduction = 0
        print(f"Reduction in Waiting Time: {reduction:.2f}%")
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(ai_waiting_times, baseline_waiting_times)
        print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("The reduction in waiting time is statistically significant (p < 0.05).")
        else:
            print("The reduction in waiting time is not statistically significant (p >= 0.05).")
        
        # Plot waiting times (sample every 10th point for readability)
        plt.figure(figsize=(12, 6))
        sample_indices = range(0, len(ai_df), 10)
        plt.plot(ai_df['time_of_day'].iloc[sample_indices], 
                ai_waiting_times[sample_indices], 
                label='AI-Based Waiting Time', alpha=0.7)
        plt.plot(baseline_df['time_of_day'].iloc[sample_indices], 
                baseline_waiting_times[sample_indices], 
                label='Baseline Waiting Time', alpha=0.7, linestyle='--')
        plt.xlabel('Time of Day')
        plt.ylabel('Waiting Time (Vehicles on Red)')
        plt.title('Waiting Time Comparison: AI vs. Baseline')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/waiting_time_comparison.png')
        plt.close()
        print(f"Waiting time comparison plot saved to {output_dir}/waiting_time_comparison.png")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    evaluate_congestion()