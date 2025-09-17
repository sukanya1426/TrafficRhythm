import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_traffic_data(input_file='four_road_traffic_data.csv', 
                          output_file='preprocessed_four_road_data.csv'):
    """
    Preprocess the traffic dataset: add features, normalize, encode categorical variables.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the preprocessed CSV file.
    """
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Convert time_of_day to minutes since midnight
    df['minutes'] = df['time_of_day'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    
    # Add peak hour flag (7-10 AM, 4-7 PM)
    df['peak_hour'] = ((df['minutes'] >= 7 * 60) & (df['minutes'] < 10 * 60) | 
                      (df['minutes'] >= 16 * 60) & (df['minutes'] < 19 * 60)).astype(int)
    
    # One-hot encode day_of_week
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='day')
    
    # Normalize numerical columns
    scaler = StandardScaler()
    cols_to_normalize = [
        'vehicle_count_north', 'vehicle_count_south', 'vehicle_count_east', 
        'vehicle_count_west', 'previous_signal_north_south', 'previous_signal_east_west',
        'minutes'
    ]
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    # Save preprocessed dataset
    df.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")

if __name__ == "__main__":
    preprocess_traffic_data()
