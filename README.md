# ⚡ AI-Based Fault Detection in Smart Grid

### Using Autoencoder-Based Anomaly Detection

---

## 📘 Overview

This project implements an **AI-based fault detection system** for smart grids using a **Deep Learning Autoencoder**. The model learns the behavior of *normal* grid operations and detects faults by identifying deviations in reconstruction error.

The system supports:

* Intelligent anomaly detection
* Visualization of fault points
* High accuracy using multi-parameter grid data
* Streamlit-based interactive dashboard

This project is ideal for smart grid automation, research, and real-world monitoring systems.

---

## 🚀 Features

✔ AI-powered unsupervised fault detection
✔ Autoencoder deep learning model
✔ Anomaly classification based on reconstruction error
✔ Visual graph plotting for faults
✔ Streamlit web interface
✔ Clean dataset preprocessing
✔ Scalable for real-time deployment

---

## 📂 Project Structure

```
fault-detection-in-smart-grid/
│── dataset.csv
│── train_model.py
│── detect_faults.py
│── app.py                      # Streamlit dashboard
│── model.h5                    # Trained Autoencoder model
│── README.md
```

---

## 🧠 Methodology

1. **Data Collection**
   Existing smart grid dataset containing 50,000 records with electrical parameters.

2. **Data Preprocessing**

   * Removed missing values
   * Normalized features using MinMaxScaler
   * Removed timestamp column
   * Converted values into ML-compatible format

3. **Model Development**

   * Built an Autoencoder architecture
   * Trained using normal grid data
   * Reconstruction error used to identify faults

4. **Fault Detection**

   * Tested on complete dataset
   * Set threshold using mean + standard deviation
   * Points above threshold → **F A U L T**

5. **Visualization**

   * Plots show error line, thresholds, and fault points

---

## 📊 Dataset Information

The dataset includes:

| Parameter        | Description             |
| ---------------- | ----------------------- |
| Voltage (V)      | Grid Voltage            |
| Current (A)      | Line Current            |
| Power (kW)       | Active Power            |
| Reactive Power   | kVAR                    |
| Power Factor     | Load Efficiency         |
| Temperature      | Environmental condition |
| Solar/Wind Input | Renewable contribution  |
| Load Indicators  | Fluctuation behavior    |

Total Records: **50,000**

Format: **CSV**

---

## 🔧 Technologies Used

| Library            | Purpose                |
| ------------------ | ---------------------- |
| TensorFlow / Keras | Autoencoder model      |
| Pandas             | Dataset handling       |
| NumPy              | Calculations           |
| Scikit-learn       | Scaling (MinMaxScaler) |
| Matplotlib         | Graph plotting         |
| Streamlit          | Web app interface      |

---

## 🛠 Installation

### 1. Clone the repository

```
git clone https://github.com/Metikarthik/fault-detection-in-smart-grid-.git
cd fault-detection-in-smart-grid-
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the model

```
python train_model.py
```

### Detect faults

```
python detect_faults.py
```

### Run Streamlit dashboard

```
streamlit run app.py
```

---

## 📈 Sample Output Graph

Shows reconstruction error, threshold line, and fault points.

---

## 🧪 Results

* Accurate detection of abnormal grid conditions
* Faults clearly identifiable in visualization
* Autoencoder adapts to nonlinear grid behavior
* Works even without labeled data

---

## 🔮 Future Enhancements

* Real-time SCADA integration
* Fault type classification
* LSTM/Transformer-based prediction
* Mobile app alert system
* Cloud/edge deployment

---

## 👨‍💻 Author

**Karthik Meti**
GitHub: [Metikarthik](https://github.com/Metikarthik)

---

## ⭐ Contribute

Feel free to fork, contribute, or open issues in the repository.




