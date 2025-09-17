import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define feature columns used throughout the model
feature_columns = [
    'cars', 'buses', 'trucks', 'rickshaws', 'bikes',
    'hour', 'minute', 'day_of_week',
    'is_rush_hour', 'is_business_hours', 'is_night', 'is_weekend',
    'avg_vehicles', 'avg_wait_time'
]

def train_model(data_file="traffic_data.csv"):
    """Train the Random Forest model with enhanced features."""
    data = pd.read_csv(data_file)
    X = data[feature_columns]
    y = data["greenTime"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=15, 
                                min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    test_score = model.score(X_test, y_test)
    print(f"✅ Model trained with enhanced features (R² Score: {test_score:.3f})")
    
    joblib.dump(model, "traffic_rf_model.pkl")
    return model

def get_example_prediction():
    """Get a prediction using example data."""
    new_state = {
        "cars": 12, "buses": 2, "trucks": 1, "rickshaws": 4, "bikes": 9,
        "hour": 18, "minute": 30, "day_of_week": 2,
        "is_rush_hour": 1, "is_business_hours": 1, "is_night": 0, "is_weekend": 0,
        "avg_vehicles": 25, "avg_wait_time": 45
    }
    
    features = [[new_state[col] for col in feature_columns]]
    predicted_time = model.predict(features)[0]
    predicted_time = max(10, min(40, int(predicted_time)))
    
    print(f"Predicted green time: {predicted_time} seconds")
    return predicted_time

def log_new_cycle(cycle_data, filename="traffic_data.csv"):
    """Log a new traffic cycle and retrain model if needed."""
    try:
        df = pd.read_csv(filename)
        new_row = pd.DataFrame([cycle_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
        print("✅ New cycle data logged successfully")
        
        if len(df) % 100 == 0:
            print("🔄 Retraining model with updated dataset...")
            global model
            model = train_model()
            print("✅ Model retrained successfully")
            
    except Exception as e:
        print("❌ Error logging cycle data:", str(e))

# Initial model training
model = train_model()

if __name__ == "__main__":
    prediction = get_example_prediction()
    print("\nExample prediction completed.")
    
    example_cycle = {
        "cars": 15, "buses": 1, "trucks": 0, "rickshaws": 3, "bikes": 7,
        "hour": 17, "minute": 45, "day_of_week": 3,
        "is_rush_hour": 1, "is_business_hours": 1, "is_night": 0, "is_weekend": 0,
        "greenTime": 25, "vehiclesCrossed": 12, "avgWaitingTime": 35,
        "avg_vehicles": 22, "avg_wait_time": 40
    }
    
    log_new_cycle(example_cycle)
