import pandas as pd

def load_weather_data():
    """
    Load the Seattle weather data from local CSV file.
    Returns a pandas DataFrame with the weather data.
    """
    try:
        # Read the downloaded CSV file
        df = pd.read_csv('data/seattle-weather.csv')
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'date': 'DATE',
            'temp_max': 'TMAX',
            'temp_min': 'TMIN',
            'precipitation': 'PRCP',
            'wind': 'WIND',
            'weather': 'WEATHER'
        })
        
        # Convert date to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Convert temperature from Celsius to Fahrenheit
        df['TMAX'] = df['TMAX'].apply(lambda x: (x * 9/5) + 32)
        df['TMIN'] = df['TMIN'].apply(lambda x: (x * 9/5) + 32)
        
        print("Data loaded successfully!")
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def verify_data():
    """Verify that the weather data was loaded correctly."""
    try:
        df = pd.read_csv('data/seattle-weather.csv')
        print("\nData verification:")
        print(f"Number of records: {len(df)}")
        print("\nFirst few records:")
        print(df.head())
        print("\nData columns:")
        print(df.columns.tolist())
        print("\nMissing values:")
        print(df.isnull().sum())
        return True
    except Exception as e:
        print(f"Error verifying data: {str(e)}")
        return False