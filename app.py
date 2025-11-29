import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------
# 1️⃣ PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Smart Grid Fault Detection", layout="wide")
st.title("⚡ Smart Grid Fault Detection Using AI")

# -------------------------------
# 2️⃣ LOAD DATA AND MODEL
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("autoencoder_model.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    df = load_data()
    model, scaler = load_model()
    st.success("✅ Model and dataset loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model or dataset: {e}")
    st.stop()

# -------------------------------
# 3️⃣ COMPLETE DATASET DISPLAY
# -------------------------------
st.subheader("📊 Complete Dataset")

st.write(f"**Total Rows:** {len(df)} | **Total Columns:** {len(df.columns)}")

# Optional pagination for large datasets
page_size = 1000
page = st.number_input("Page", 1, (len(df) // page_size) + 1, 1)
start = (page - 1) * page_size
end = start + page_size
st.dataframe(df.iloc[start:end], use_container_width=True, height=500)

# Download complete dataset
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Complete Dataset (CSV)",
    data=csv_data,
    file_name="complete_dataset.csv",
    mime="text/csv",
)

# -------------------------------
# 4️⃣ FAULT DETECTION LOGIC
# -------------------------------
st.subheader("🔍 Fault Detection Analysis")

numeric_df = df.select_dtypes(include=[np.number])
scaled_data = scaler.transform(numeric_df)
reconstructed = model.predict(scaled_data)
mse = np.mean(np.power(scaled_data - reconstructed, 2), axis=1)

# Threshold based on 95th percentile
threshold = np.percentile(mse, 95)
faults = mse > threshold
df["Fault Detected"] = faults

num_faults = faults.sum()
st.metric(label="Total Faults Detected", value=int(num_faults))
st.metric(label="Total Samples", value=len(df))

# -------------------------------
# 5️⃣ GRAPH VISUALIZATION
# -------------------------------
st.subheader("📈 Fault Detection Graph")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mse, label="Reconstruction Error", color="blue")
ax.axhline(y=threshold, color="red", linestyle="--", label="Fault Threshold")

# Highlight fault points
fault_indices = np.where(faults)[0]
ax.scatter(fault_indices, mse[fault_indices], color="red", label="Faults", s=15)

ax.set_title("Fault Detection Graph")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Reconstruction Error")
ax.legend()
st.pyplot(fig)

# -------------------------------
# 6️⃣ FAULT LIST DISPLAY
# -------------------------------
st.subheader("⚠️ Detected Faults")

fault_df = df[faults]
st.write(f"**Total Fault Rows:** {len(fault_df)}")
st.dataframe(fault_df, use_container_width=True, height=500)

# Download faults
fault_csv = fault_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Faults (CSV)",
    data=fault_csv,
    file_name="faults_detected.csv",
    mime="text/csv",
)

