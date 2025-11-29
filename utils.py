# utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path='dataset.csv'):
    """Load dataset and scale numeric features"""
    df = pd.read_csv(path)

    features = [
        'Voltage (V)', 'Current (A)', 'Power Consumption (kW)',
        'Reactive Power (kVAR)', 'Power Factor', 'Solar Power (kW)',
        'Wind Power (kW)', 'Grid Supply (kW)', 'Voltage Fluctuation (%)',
        'Temperature (°C)', 'Humidity (%)', 'Electricity Price (USD/kWh)',
        'Predicted Load (kW)'
    ]

    # Handle missing values (if any)
    df = df.fillna(df.mean(numeric_only=True))

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, scaler
