import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import joblib

def load_data(path):
    print(f"📂 Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("🔹 Columns:", df.columns.tolist())

    # Drop timestamp or non-numeric columns
    df = df.select_dtypes(include=[np.number])
    print(f"🔢 Numeric columns used: {df.shape[1]}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    joblib.dump(scaler, 'scaler.joblib')
    print("💾 Saved scaler as scaler.joblib")
    return X

def build_autoencoder(input_dim):
    print("🧠 Building autoencoder model...")
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    print("✅ Model built successfully")
    return model

if __name__ == "__main__":
    print("🚀 Starting Smart Grid AI model training...")
    X = load_data('dataset.csv')
    model = build_autoencoder(X.shape[1])
    print("📈 Training started...")
    history = model.fit(
        X, X,
        epochs=1,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    model.save('autoencoder_model.keras')
    print("✅ Training complete! Model saved as autoencoder_model.keras")
