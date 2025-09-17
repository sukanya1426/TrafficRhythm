import numpy as np
import pandas as pd
from traffic_env import FourRoadTrafficEnv
import joblib

def discretize_state(state, bins=10):
    """
    Discretize vehicle counts to reduce state space.
    
    Args:
        state (list): [north, south, east, west, green_time].
        bins (int): Number of bins for vehicle counts.
    
    Returns:
        tuple: Discretized state indices.
    """
    vehicle_bins = np.linspace(-3, 3, bins)  # Normalized range from Day 1 preprocessing
    state_idx = [
        np.digitize(state[0], vehicle_bins),
        np.digitize(state[1], vehicle_bins),
        np.digitize(state[2], vehicle_bins),
        np.digitize(state[3], vehicle_bins)
    ]
    return tuple(np.clip(state_idx, 0, bins-1))

def train_rl_model(data_file='preprocessed_four_road_data.csv', q_table_file='q_table.pkl', episodes=1000):
    """
    Train a Q-learning agent for traffic signal optimization.
    
    Args:
        data_file (str): Path to preprocessed traffic data CSV.
        q_table_file (str): Path to save the Q-table.
        episodes (int): Number of training episodes.
    """
    env = FourRoadTrafficEnv(data_file)
    q_table = np.zeros((10, 10, 10, 10, len(env.actions)))  # 10 bins per vehicle count
    alpha, gamma, epsilon = 0.1, 0.9, 0.1
    
    for episode in range(episodes):
        state = env.reset()
        state_idx = discretize_state(state)
        done = False
        total_reward = 0
        
        while not done:
            # Choose action (epsilon-greedy)
            if np.random.random() < epsilon:
                action = np.random.randint(len(env.actions))
            else:
                action = np.argmax(q_table[state_idx])
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state_idx = discretize_state(next_state)
            
            # Update Q-table
            q_table[state_idx][action] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx][action]
            )
            
            state_idx = next_state_idx
            total_reward += reward
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
    
    # Save Q-table
    joblib.dump(q_table, q_table_file)
    print(f"Q-table saved to {q_table_file}")

if __name__ == "__main__":
    train_rl_model()