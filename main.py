import os
import pandas as pd
from src.data_processing import clean_data, prepare_features
from src.model import train_model, evaluate_model, make_prediction
from src.utils import load_weather_data, verify_data
from sklearn.model_selection import train_test_split

def predict_temperature(model, scaler):
    """Function to get user input and make predictions"""
    try:
        print("\nEnter weather data for prediction:")
        date = input("Date (YYYY-MM-DD): ")
        precipitation = float(input("Precipitation (mm): "))
        wind = float(input("Wind speed (m/s): "))
        min_temp = float(input("Minimum temperature (°F): "))
        weather = input("Weather type (sun, rain, drizzle, snow, fog): ").lower()
        
        # Validate weather type
        valid_weather_types = ['sun', 'rain', 'drizzle', 'snow', 'fog']
        if weather not in valid_weather_types:
            raise ValueError(f"Weather type must be one of: {', '.join(valid_weather_types)}")
        
        # Create a DataFrame with the input data
        new_data = pd.DataFrame({
            'DATE': [date],
            'PRCP': [precipitation],
            'WIND': [wind],
            'TMIN': [min_temp],
            'WEATHER': [weather]
        })
        
        # Make prediction
        predicted_temp = make_prediction(model, scaler, new_data)
        
        print(f"\nPredicted maximum temperature: {predicted_temp:.2f}°F")
        
    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

def main():
    try:
        # Verify the data file exists
        if not os.path.exists('data/seattle-weather.csv'):
            print("Error: seattle-weather.csv not found in data directory!")
            return
            
        # Load data
        print("Loading weather data...")
        df = load_weather_data()
        
        if df is None:
            print("Failed to load data!")
            return
            
        # Verify the data
        if not verify_data():
            print("Data verification failed. Please check the data.")
            return
        
        # Clean data
        print("Cleaning data...")
        df_cleaned = clean_data(df)
        
        # Prepare features
        print("Preparing features...")
        X, y, scaler = prepare_features(df_cleaned)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Interactive prediction loop
        while True:
            predict_temperature(model, scaler)
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()