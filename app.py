import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="OncoPredict: Advanced Tumor Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1a3c61;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Data & Model Loading ---
@st.cache_resource
def load_scaler_and_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Replicate exact split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    model = joblib.load("Disease_prediction_model.joblib")
    
    # Compute population averages for radar chart
    df_all = X.copy()
    df_all['target'] = y
    mean_benign = df_all[df_all['target'] == 1].drop('target', axis=1).mean()
    mean_malignant = df_all[df_all['target'] == 0].drop('target', axis=1).mean()
    
    return scaler, model, data.feature_names, X, mean_benign, mean_malignant

scaler, model, feature_names, X_df, mean_benign, mean_malignant = load_scaler_and_model()


# --- Sidebar: Inputs ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3003/3003306.png", width=100)
st.sidebar.title("Patient Profile")
st.sidebar.markdown("Define the medical attributes for the tumor diagnostic modeling.")

# Group features
mean_features = [f for f in feature_names if 'mean' in f]
error_features = [f for f in feature_names if 'error' in f]
worst_features = [f for f in feature_names if 'worst' in f]

input_data = {}

def render_inputs(features, group_name, icon, expanded=False):
    with st.sidebar.expander(f"{icon} {group_name}", expanded=expanded):
        for feature in features:
            min_val = float(X_df[feature].min())
            max_val = float(X_df[feature].max())
            mean_val = float(X_df[feature].mean())
            input_data[feature] = st.slider(
                feature.replace("mean ", "").capitalize(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100.0,
                key=feature
            )

render_inputs(mean_features, "Mean Features", "📊", expanded=True)
render_inputs(error_features, "Error Features", "📉")
render_inputs(worst_features, "Worst Features", "📈")


# --- Main App Header ---
st.title("🧬 OncoPredict: Advanced Tumor Diagnostics")
st.markdown("""
Welcome to the advanced predictive modeling portal. This interface uses an optimized **Random Forest Classifier** to analyze 30 cellular metrics from digitized fine needle aspirate (FNA) images of breast masses, predicting morphological classifications in real-time.
""")
st.divider()

# --- Predictions logic ---
input_df = pd.DataFrame([input_data])[feature_names]
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

# Class mapping (0 = Malignant, 1 = Benign)
is_malignant = prediction == 0
pred_label = "Malignant" if is_malignant else "Benign"
pred_prob = probabilities[0] if is_malignant else probabilities[1]
pred_color = "red" if is_malignant else "green"

# --- Top Row: Results & Gauge ---
col1, col2 = st.columns([1.2, 2])

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Diagnostic Assessment")
    if is_malignant:
        st.error(f"### 🚨 {pred_label}\nHigh risk indicators detected. Immediate consultation recommended.")
    else:
        st.success(f"### ✅ {pred_label}\nMetrics align with benign morphologies.")
    
    st.markdown(f"**Confidence Score:** `{pred_prob*100:.1f}%`")
    
    # Download Report button
    report_content = f"--- OncoPredict Diagnostic Report ---\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += f"PREDICTION: {pred_label}\n"
    report_content += f"CONFIDENCE: {pred_prob*100:.1f}%\n\n-- PATIENT METRICS --\n"
    for k, v in input_data.items():
        report_content += f"{k}: {v:.4f}\n"
        
    st.download_button(
        label="📄 Download Full Diagnostic Report",
        data=report_content,
        file_name=f"OncoPredict_Report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Index ({pred_label})", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': pred_color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': pred_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f2f6'},
                {'range': [50, 80], 'color': '#e1e5ee'},
                {'range': [80, 100], 'color': '#cdd4e0'}],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- Middle Row: Radar & Feature Importance ---
st.divider()
st.subheader("Data Analytics & Model Insights")
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("#### Patient Biomarkers vs. Demographics")
    # Radar chart for the first 10 'mean' features
    radar_features = mean_features[:10]
    
    # Scale for radar chart to display harmonized geometric shape
    radar_benign = scaler.transform(pd.DataFrame([mean_benign])[feature_names])[0]
    radar_malignant = scaler.transform(pd.DataFrame([mean_malignant])[feature_names])[0]
    
    idx = [feature_names.tolist().index(f) for f in radar_features]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_benign[idx],
        theta=radar_features,
        fill='none',
        name='Avg Benign Profile',
        line_color='green',
        opacity=0.6
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_malignant[idx],
        theta=radar_features,
        fill='none',
        name='Avg Malignant Profile',
        line_color='red',
        opacity=0.6
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=input_scaled[0][idx],
        theta=radar_features,
        fill='toself',
        name='Current Patient',
        line_color='blue',
        fillcolor='rgba(0,0,255,0.2)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[-3, 5])
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("#### Random Forest Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10 features
    
    top_features = [feature_names[i].replace("worst ", "w_").replace("mean ", "m_") for i in indices]
    top_importances = importances[indices]
    
    fig_bar = px.bar(
        x=top_importances, 
        y=top_features, 
        orientation='h',
        labels={'x': 'Relative Importance', 'y': ''},
        color=top_importances,
        color_continuous_scale="Blues"
    )
    fig_bar.update_layout(
        yaxis={'categoryorder':'total ascending'}, 
        height=400, 
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
