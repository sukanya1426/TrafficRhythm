import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(input_file='preprocessed_four_road_data.csv', model_file='rf_model.pkl'):
    """
    Train a tuned Random Forest Classifier to predict congestion levels.
    
    Args:
        input_file (str): Path to the preprocessed CSV file.
        model_file (str): Path to save the trained model.
    """
    # Load preprocessed data
    df = pd.read_csv(input_file)
    
    # Prepare features (X) and target (y)
    X = df.drop(['congestion_level', 'time_of_day'], axis=1)
    y = df['congestion_level']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and tune Random Forest model
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Save the trained model
    joblib.dump(best_model, model_file)
    print(f"Trained model saved to {model_file}")

if __name__ == "__main__":
    train_model()