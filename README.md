# 🧬 OncoPredict: Advanced Tumor Diagnostics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://oncopredict-advanced-tumor-diagnostics.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**OncoPredict** is a state-of-the-art predictive modeling portal designed for clinical researchers and healthcare professionals. Using an optimized **Random Forest Classifier**, it analyzes 30 cellular metrics from digitized fine needle aspirate (FNA) images of breast masses to provide real-time morphological classifications.

---

## ✨ Key Features

- **🔍 Real-time Diagnostics**: Instant prediction of tumor morphology (Benign vs. Malignant).
- **📊 Interactive Analytics**: 
  - **Confidence Index Gauge**: Visualizes the model's certainty.
  - **Biomarker Radar Charts**: Compares patient data against population averages for Benign and Malignant profiles.
  - **Feature Importance**: Shows which metrics (e.g., area, concavity) are driving the prediction.
- **📄 Diagnostic Reports**: Generate and download a structured text report containing all patient metrics and predictions.
- **🎨 Premium UI/UX**: Built with a clean, medical-grade aesthetic using Streamlit and custom CSS.

## 🛠️ Tech Stack

- **Core**: Python 3.9
- **ML Framework**: Scikit-Learn (Random Forest Classifier)
- **Frontend**: Streamlit
- **Visualization**: Plotly (Radar charts, Gauges, Bar charts)
- **Data Handling**: Pandas, NumPy
- **Model Persistence**: Joblib

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aneela123-Lab/OncoPredict-Advanced-Tumor-Diagnostics.git
   cd OncoPredict-Advanced-Tumor-Diagnostics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit application
├── Disease_prediction_model.joblib # Trained Random Forest model
├── requirements.txt             # Project dependencies
├── german_credit_data.csv       # Dataset (Sample reference)
└── README.md                    # Project documentation
```

## 📊 Model Performance

The underlying model is trained on the **UCI Breast Cancer Wisconsin (Diagnostic) Dataset**, achieving:
- **Accuracy**: ~96%
- **Precision**: 95%
- **Recall**: 97%

## 🤝 Contributing

Contributions to OncoPredict are welcome! If you have suggestions for new features or improvements, feel free to:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Disclaimer**: *This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
