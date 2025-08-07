import numpy as np
import pandas as pd

class FourRoadTrafficEnv:
    """
    RL environment for a four-road intersection traffic signal optimization.
    """
    def __init__(self, data_file='preprocessed_four_road_data.csv'):
        """
        Initialize the environment with preprocessed traffic data.
        
        Args:
            data_file (str): Path to preprocessed traffic data CSV.
        """
        self.data = pd.read_csv(data_file)
        self.actions = [(20, 0), (30, 0), (40, 0), (0, 20), (0, 30), (0, 40)]  
        self.state = None
        self.current_step = 0
        self.max_steps = len(self.data)
        self.reset()
    
    def reset(self):
        """
        Reset the environment to a random state from the dataset.
        
        Returns:
            list: Initial state [north, south, east, west, green_time].
        """
        self.current_step = np.random.randint(0, self.max_steps)
        self.state = [
            self.data['vehicle_count_north'].iloc[self.current_step],
            self.data['vehicle_count_south'].iloc[self.current_step],
            self.data['vehicle_count_east'].iloc[self.current_step],
            self.data['vehicle_count_west'].iloc[self.current_step],
            20  # Default green time
        ]
        return self.state
    
    def step(self, action):
        """
        Take an action and update the environment.
        
        Args:
            action (int): Index of action (green time for NS or EW).
        
        Returns:
            tuple: (next_state, reward, done, info).
        """
        green_ns, green_ew = self.actions[action]
        self.state[4] = green_ns if green_ns else green_ew
        
        # Calculate waiting time (vehicles on red lights)
        if green_ns:  # North-South green, East-West red
            waiting_time = self.state[2] + self.state[3]  # East + West
        else:  # East-West green, North-South red
            waiting_time = self.state[0] + self.state[1]  # North + South
        reward = -waiting_time
        
        # Move to next time step
        self.current_step = (self.current_step + 1) % self.max_steps
        self.state[:4] = [
            self.data['vehicle_count_north'].iloc[self.current_step],
            self.data['vehicle_count_south'].iloc[self.current_step],
            self.data['vehicle_count_east'].iloc[self.current_step],
            self.data['vehicle_count_west'].iloc[self.current_step]
        ]
        
        done = self.current_step >= self.max_steps - 1
        info = {}
        return self.state, reward, done, info