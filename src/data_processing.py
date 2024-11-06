import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(df):
    """Clean the Seattle weather dataset."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Create temporal features
    df['Month'] = df['DATE'].dt.month
    df['Day'] = df['DATE'].dt.day
    df['Year'] = df['DATE'].dt.year
    
    # Create weather type dummy variables
    weather_dummies = pd.get_dummies(df['WEATHER'], prefix='weather')
    df = pd.concat([df, weather_dummies], axis=1)
    
    # Select relevant features
    feature_columns = ['Month', 'Day', 'Year', 'PRCP', 'WIND', 'TMIN'] + \
                     [col for col in df.columns if col.startswith('weather_')]
    
    df = df[feature_columns + ['TMAX']]  # Keep TMAX as target
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    # Split features and target
    X = df.drop('TMAX', axis=1)
    y = df['TMAX']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler