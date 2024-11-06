from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd

def train_model(X_train, y_train):
    """Train the weather prediction model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2
    }

def make_prediction(model, scaler, new_data):
    """
    Make temperature predictions for new weather data.
    
    Parameters:
    - model: Trained RandomForestRegressor model
    - scaler: Fitted StandardScaler
    - new_data: DataFrame with columns: DATE, PRCP, WIND, TMIN, WEATHER
    
    Returns:
    - Predicted maximum temperature in Fahrenheit
    """
    # Process the new data similar to training data
    new_data = new_data.copy()
    
    # Create temporal features
    new_data['Month'] = pd.to_datetime(new_data['DATE']).dt.month
    new_data['Day'] = pd.to_datetime(new_data['DATE']).dt.day
    new_data['Year'] = pd.to_datetime(new_data['DATE']).dt.year
    
    # Create weather dummy variables
    weather_dummies = pd.get_dummies(new_data['WEATHER'], prefix='weather')
    new_data = pd.concat([new_data, weather_dummies], axis=1)
    
    # Select the same features used in training
    feature_columns = ['Month', 'Day', 'Year', 'PRCP', 'WIND', 'TMIN'] + \
                     [col for col in new_data.columns if col.startswith('weather_')]
    
    # Ensure all weather types from training are present
    missing_columns = set(scaler.feature_names_in_) - set(feature_columns)
    for col in missing_columns:
        new_data[col] = 0
        
    # Select and order columns to match training data
    X_new = new_data[scaler.feature_names_in_]
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)
    
    return prediction[0]  # Return the predicted temperature