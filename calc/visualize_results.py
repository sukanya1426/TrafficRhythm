import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(supervised_file='predicted_signal_timings.csv', 
                     rl_file='rl_signal_timings.csv', 
                     output_dir='plots'):
    """
    Visualize vehicle counts and green times for North-South and East-West directions.
    
    Args:
        supervised_file (str): Path to supervised model output CSV.
        rl_file (str): Path to RL model output CSV.
        output_dir (str): Directory to save plots.
    """
    # Load data
    supervised_df = pd.read_csv(supervised_file)
    rl_df = pd.read_csv(rl_file)
    
    # Convert time_of_day to minutes for plotting
    supervised_df['minutes'] = supervised_df['time_of_day'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    rl_df['minutes'] = rl_df['time_of_day'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    
    # Plot 1: Vehicle Counts
    plt.figure(figsize=(12, 6))
    plt.plot(supervised_df['minutes'], supervised_df['vehicle_count_north'], label='North Vehicles', alpha=0.7)
    plt.plot(supervised_df['minutes'], supervised_df['vehicle_count_south'], label='South Vehicles', alpha=0.7)
    plt.plot(supervised_df['minutes'], supervised_df['vehicle_count_east'], label='East Vehicles', alpha=0.7)
    plt.plot(supervised_df['minutes'], supervised_df['vehicle_count_west'], label='West Vehicles', alpha=0.7)
    plt.xlabel('Time (Minutes since Midnight)')
    plt.ylabel('Vehicle Count (Normalized)')
    plt.title('Vehicle Counts Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/vehicle_counts.png')
    plt.close()
    
    # Plot 2: Supervised vs. RL Green Times (North-South)
    plt.figure(figsize=(12, 6))
    plt.plot(supervised_df['minutes'], supervised_df['green_time_north_south'], 
             label='Supervised NS Green Time', alpha=0.7)
    plt.plot(rl_df['minutes'], rl_df['green_time_north_south'], 
             label='RL NS Green Time', alpha=0.7, linestyle='--')
    plt.xlabel('Time (Minutes since Midnight)')
    plt.ylabel('Green Time (Seconds)')
    plt.title('North-South Green Times: Supervised vs. RL')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/ns_green_times.png')
    plt.close()
    
    # Plot 3: Supervised vs. RL Green Times (East-West)
    plt.figure(figsize=(12, 6))
    plt.plot(supervised_df['minutes'], supervised_df['green_time_east_west'], 
             label='Supervised EW Green Time', alpha=0.7)
    plt.plot(rl_df['minutes'], rl_df['green_time_east_west'], 
             label='RL EW Green Time', alpha=0.7, linestyle='--')
    plt.xlabel('Time (Minutes since Midnight)')
    plt.ylabel('Green Time (Seconds)')
    plt.title('East-West Green Times: Supervised vs. RL')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/ew_green_times.png')
    plt.close()
    
    print(f"Plots saved to {output_dir}/: vehicle_counts.png, ns_green_times.png, ew_green_times.png")

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    visualize_results()