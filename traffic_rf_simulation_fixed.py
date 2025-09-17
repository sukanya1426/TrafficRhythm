import pandas as pd
import numpy as np
import pygame
import threading
import time
import random
import os
import csv
from datetime import datetime
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
model = load('traffic_rf_model.pkl')

# Global variables
defaultMinimum = 10
defaultMaximum = 40

# Vehicle types and directions
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Stop lines for each direction
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}

# Vehicle speeds
speeds = {'car':2.5, 'bus':2, 'truck':2, 'rickshaw':2.5, 'bike':3}

# Coordinates
signalCoods = [(530,230), (810,230), (810,570), (530,570)]

# Starting coordinates for vehicles
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        path = f"images/{direction}/{vehicleClass}.png"
        self.currentImage = pygame.image.load(path)
        self.speed = speeds[vehicleClass]
    
    def move(self, current_green, current_yellow):
        should_stop = False
        
        if self.direction == 'right':
            if not self.crossed and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
            if not self.crossed and current_green != 0 and current_yellow == 0:
                if self.x + self.currentImage.get_rect().width > stopLines[self.direction] - 20:
                    should_stop = True
            if not should_stop:
                self.x += self.speed
        
        elif self.direction == 'left':
            if not self.crossed and self.x < stopLines[self.direction]:
                self.crossed = 1
            if not self.crossed and current_green != 2 and current_yellow == 0:
                if self.x < stopLines[self.direction] + 20:
                    should_stop = True
            if not should_stop:
                self.x -= self.speed
        
        elif self.direction == 'up':
            if not self.crossed and self.y < stopLines[self.direction]:
                self.crossed = 1
            if not self.crossed and current_green != 3 and current_yellow == 0:
                if self.y < stopLines[self.direction] + 20:
                    should_stop = True
            if not should_stop:
                self.y -= self.speed
        
        elif self.direction == 'down':
            if not self.crossed and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
            if not self.crossed and current_green != 1 and current_yellow == 0:
                if self.y + self.currentImage.get_rect().height > stopLines[self.direction] - 20:
                    should_stop = True
            if not should_stop:
                self.y += self.speed
        
        # Remove vehicle if it goes off screen
        if (self.direction == 'right' and self.x > 1400 or
            self.direction == 'left' and self.x < 0 or
            self.direction == 'up' and self.y < 0 or
            self.direction == 'down' and self.y > 800):
            return True
        return False

def get_time_features(timeOfDay=None):
    current_time = datetime.now()
    hour = timeOfDay if timeOfDay is not None else current_time.hour
    minute = current_time.minute
    day_of_week = current_time.weekday()
    
    morning_peak = 1 if 7 <= hour <= 10 else 0
    evening_peak = 1 if 16 <= hour <= 19 else 0
    is_rush_hour = morning_peak or evening_peak
    is_business_hours = 1 if 9 <= hour <= 17 else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return {
        'hour': hour,
        'minute': minute,
        'day_of_week': day_of_week,
        'is_rush_hour': is_rush_hour,
        'is_business_hours': is_business_hours,
        'is_night': is_night,
        'is_weekend': is_weekend
    }

class TrafficSimulationRF:
    def __init__(self):
        pygame.init()
        self.setup_display()
        
        self.vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 
                        'down': {0:[], 1:[], 2:[], 'crossed':0}, 
                        'left': {0:[], 1:[], 2:[], 'crossed':0}, 
                        'up': {0:[], 1:[], 2:[], 'crossed':0}}
        self.currentGreen = 0
        self.nextGreen = 1
        self.currentYellow = 0
        self.greenTime = 20
        self.yellowTime = 5
        self.rf_model = model
        
        self.start_threads()
    
    def setup_display(self):
        self.screenWidth = 1400
        self.screenHeight = 800
        self.screenSize = (self.screenWidth, self.screenHeight)
        self.background = pygame.image.load('images/mod_int.png')
        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("Traffic Simulation with ML")
        
        self.redSignal = pygame.image.load('images/signals/red.png')
        self.yellowSignal = pygame.image.load('images/signals/yellow.png')
        self.greenSignal = pygame.image.load('images/signals/green.png')
        self.font = pygame.font.Font(None, 30)
    
    def getCurrentFeatures(self):
        try:
            cars, buses, trucks, rickshaws, bikes = 0, 0, 0, 0, 0
            direction = directionNumbers[self.nextGreen]
            
            for lane in self.vehicles[direction].values():
                if isinstance(lane, list):
                    for vehicle in lane:
                        if vehicle.vehicleClass == 'car': cars += 1
                        elif vehicle.vehicleClass == 'bus': buses += 1
                        elif vehicle.vehicleClass == 'truck': trucks += 1
                        elif vehicle.vehicleClass == 'rickshaw': rickshaws += 1
                        elif vehicle.vehicleClass == 'bike': bikes += 1
            
            return {
                'cars': cars,
                'buses': buses,
                'trucks': trucks,
                'rickshaws': rickshaws,
                'bikes': bikes,
                'timeOfDay': datetime.now().hour
            }
            
        except Exception as e:
            print(f"Error getting features: {str(e)}")
            return {
                'cars': 0, 'buses': 0, 'trucks': 0, 'rickshaws': 0, 'bikes': 0,
                'timeOfDay': datetime.now().hour
            }
    
    def predict_green_time(self):
        try:
            # Get current state
            state = self.getCurrentFeatures()
            
            # Create initial feature vector with base columns
            feature_columns = ['cars', 'buses', 'trucks', 'rickshaws', 'bikes', 'timeOfDay']
            features = pd.DataFrame([state])[feature_columns]
            
            # Calculate total volume
            features['total_volume'] = features[['cars', 'buses', 'trucks', 'rickshaws', 'bikes']].sum(axis=1)
            
            # Calculate vehicle type ratios in same order as training
            for col in ['cars', 'buses', 'trucks', 'rickshaws', 'bikes']:
                features[f'{col}_ratio'] = features[col] / features['total_volume'].where(features['total_volume'] > 0, 1)
            
            # Add time-based features
            time_features = pd.DataFrame([get_time_features(state['timeOfDay'])])
            for col in ['is_rush_hour', 'is_business_hours', 'is_night', 'is_weekend']:
                features[col] = time_features[col]
            
            # Add traffic metrics in same order as training
            features['avg_green_time'] = 20  # Default average
            features['green_time_std'] = 5   # Default standard deviation
            features['vehicles_per_second'] = features['total_volume'] / 30
            features['efficiency'] = features['total_volume'] / 20
            
            # Ensure exact column order matches training
            feature_order = (feature_columns + ['total_volume'] + 
                           [f'{col}_ratio' for col in ['cars', 'buses', 'trucks', 'rickshaws', 'bikes']] +
                           ['is_rush_hour', 'is_business_hours', 'is_night', 'is_weekend',
                            'avg_green_time', 'green_time_std', 'vehicles_per_second', 'efficiency'])
            features = features[feature_order]
            
            # Make prediction
            predicted_time = self.rf_model.predict(features)[0]
            predicted_time = max(defaultMinimum, min(defaultMaximum, int(predicted_time)))
            
            return predicted_time
            
        except Exception as e:
            print(f"❌ Error predicting green time: {str(e)}")
            return 20  # Default fallback value
    
    def start_threads(self):
        thread_generate = threading.Thread(target=self.generateVehicles)
        thread_timing = threading.Thread(target=self.updateSignals)
        thread_generate.daemon = True
        thread_timing.daemon = True
        thread_generate.start()
        thread_timing.start()
    
    def generateVehicles(self):
        while True:
            vehicle_type = random.randint(0,4)
            lane_number = 0 if vehicle_type == 4 else random.randint(1,2)
            
            direction_number = random.randint(0,3)
            direction = directionNumbers[direction_number]
            
            vehicle = Vehicle(lane_number, vehicleTypes[vehicle_type], 
                            direction_number, direction, random.randint(0,1))
            
            self.vehicles[direction][lane_number].append(vehicle)
            
            print(f"Created {vehicleTypes[vehicle_type]} in lane {lane_number} going {direction}")
            time.sleep(2)
    
    def updateSignals(self):
        while True:
            try:
                state = self.getCurrentFeatures()
                predicted_time = self.predict_green_time()
                
                print(f"\nCurrent Traffic State:")
                print(f"Direction: {directionNumbers[self.nextGreen]}")
                print(f"Vehicle Counts: Cars={state['cars']}, Buses={state['buses']}, "
                      f"Trucks={state['trucks']}, Rickshaws={state['rickshaws']}, "
                      f"Bikes={state['bikes']}")
                print(f"Predicted Green Time: {predicted_time:.1f} seconds")
                
                # Yellow phase
                self.currentYellow = 1
                time.sleep(self.yellowTime)
                self.currentYellow = 0
                
                # Green phase
                self.currentGreen = self.nextGreen
                self.nextGreen = (self.currentGreen + 1) % 4
                
                time.sleep(predicted_time)
                
            except Exception as e:
                print(f"Error in signal update: {str(e)}")
                time.sleep(20)
    
    def update_vehicles(self):
        for direction in self.vehicles:
            for lane in self.vehicles[direction]:
                if isinstance(self.vehicles[direction][lane], list):
                    self.vehicles[direction][lane] = [
                        vehicle for vehicle in self.vehicles[direction][lane]
                        if not vehicle.move(self.currentGreen, self.currentYellow)
                    ]
    
    def update_signals(self):
        for i in range(4):
            if i != self.currentGreen and not (i == self.nextGreen and self.currentYellow == 1):
                self.screen.blit(self.redSignal, signalCoods[i])
            elif i == self.nextGreen and self.currentYellow == 1:
                self.screen.blit(self.yellowSignal, signalCoods[i])
            else:
                self.screen.blit(self.greenSignal, signalCoods[i])
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.screen.blit(self.background, (0, 0))
            self.update_vehicles()
            
            for direction in self.vehicles:
                for lane in self.vehicles[direction]:
                    if isinstance(self.vehicles[direction][lane], list):
                        for vehicle in self.vehicles[direction][lane]:
                            self.screen.blit(vehicle.currentImage, (vehicle.x, vehicle.y))
            
            self.update_signals()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    simulation = TrafficSimulationRF()
    simulation.run()
