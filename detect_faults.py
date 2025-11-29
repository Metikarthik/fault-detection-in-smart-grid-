import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
import matplotlib.pyplot as plt

def detect_faults():
    print("🚀 Starting Smart Grid Fault Detection...")

    start_time = time.time()

    # Load dataset
    try:
        df = pd.read_csv('dataset.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ dataset.csv not found. Please place it in the same folder.")
        return

    # Select numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("❌ No numeric data found in dataset.")
        return
    print(f"🔢 Numeric columns used: {list(df_numeric.columns)}")

    # Load model and scaler
    print("📦 Loading trained model and scaler...")
    model = load_model('autoencoder_model.keras')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model and scaler loaded successfully!")

    # Scale data
    X = scaler.transform(df_numeric)

    # Predict reconstruction
    print("⚙️  Running reconstruction...")
    reconstructed = model.predict(X, verbose=0)

    # Compute reconstruction error
    mse = np.mean(np.power(X - reconstructed, 2), axis=1)

    # Define threshold
    threshold = np.mean(mse) + 3 * np.std(mse)
    print(f"📏 Fault threshold: {threshold:.6f}")

    # Detect faults
    fault_indices = np.where(mse > threshold)[0]
    print(f"⚡ Faults detected: {len(fault_indices)}")

    if len(fault_indices) > 0:
        faults = df.iloc[fault_indices]
        faults.to_csv('faults_detected.csv', index=False)
        print("💾 Saved detected faults to 'faults_detected.csv'")
    else:
        print("✅ No significant faults detected!")

    # Visualization
    print("📊 Generating reconstruction error plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(mse, label='Reconstruction Error', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Fault Threshold')
    plt.scatter(fault_indices, mse[fault_indices], color='orange', label='Detected Faults')
    plt.title("Smart Grid Fault Detection - Reconstruction Error")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend()
    plt.tight_layout()

    # Save plot image
    plt.savefig("fault_plot.png", dpi=300)
    print("🖼️  Saved plot image as 'fault_plot.png'")

    # Show plot
    plt.show()

    print(f"⏱️ Total detection time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    detect_faults()
