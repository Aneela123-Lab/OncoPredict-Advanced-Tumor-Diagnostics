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
    /* Dark Theme Base */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Glowing Title */
    .glow-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem !important;
        text-align: center;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        margin-bottom: 2rem;
        padding-top: 1rem;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Styled Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.6);
    }
    
    /* Text Styles */
    h1, h2, h3 { color: #f1f5f9 !important; }
    p, label { color: #94a3b8 !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
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
st.markdown("<h1 class='glow-title'>🧬 OncoPredict: Advanced Tumor Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem; margin-top: -1rem; color: #cbd5e1;'>
    Optimized Predictive Analytics for Clinical Oncology
</p>
""", unsafe_allow_html=True)
st.divider()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Dashboard", "🧠 Model Intelligence", "🔬 About OncoPredict"])

with tab1:
    # --- Predictions logic ---
    input_df = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Class mapping (0 = Malignant, 1 = Benign)
    is_malignant = prediction == 0
    pred_label = "Malignant" if is_malignant else "Benign"
    pred_prob = probabilities[0] if is_malignant else probabilities[1]
    pred_color = "#ef4444" if is_malignant else "#10b981"

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
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 255, 255, 0.05)'},
                    {'range': [50, 80], 'color': 'rgba(255, 255, 255, 0.1)'},
                    {'range': [80, 100], 'color': 'rgba(255, 255, 255, 0.15)'}],
                'threshold': {
                    'line': {'color': "#f8fafc", 'width': 3},
                    'thickness': 0.75,
                    'value': 90}
            }
        ))
        fig_gauge.update_layout(
            height=280, 
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc"},
            template='plotly_dark'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Data Analytics & Model Insights")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### Patient Biomarkers vs. Demographics")
        radar_features = mean_features[:10]
        radar_benign = scaler.transform(pd.DataFrame([mean_benign])[feature_names])[0]
        radar_malignant = scaler.transform(pd.DataFrame([mean_malignant])[feature_names])[0]
        idx = [feature_names.tolist().index(f) for f in radar_features]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_benign[idx], theta=radar_features, fill='none', name='Avg Benign Profile', line_color='#10b981', opacity=0.6))
        fig_radar.add_trace(go.Scatterpolar(r=radar_malignant[idx], theta=radar_features, fill='none', name='Avg Malignant Profile', line_color='#ef4444', opacity=0.6))
        fig_radar.add_trace(go.Scatterpolar(r=input_scaled[0][idx], theta=radar_features, fill='toself', name='Current Patient', line_color='#38bdf8', fillcolor='rgba(56, 189, 248, 0.2)'))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[-3, 5]), bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=400,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc"},
            template='plotly_dark'
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### Random Forest Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i].replace("worst ", "w_").replace("mean ", "m_") for i in indices]
        top_importances = importances[indices]
        
        fig_bar = px.bar(x=top_importances, y=top_features, orientation='h', color=top_importances, color_continuous_scale="Blues")
        fig_bar.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
            height=400, 
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc"},
            template='plotly_dark'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Model Decision Logic")
    st.markdown("""
    The system utilizes a **Random Forest Ensemble** with 100+ decision trees. 
    Each tree votes on the classification based on thresholds learned during training.
    """)
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Data Source:** UCI Breast Cancer Wisconsin (Diagnostic)")
        st.info("**Model Type:** Random Forest Classifier")
    with col_b:
        st.info("**Training Samples:** 455 Patients")
        st.info("**Test Accuracy:** 96.5%")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.header("About the Clinical Portal")
    st.write("""
    This portal is designed to provide high-fidelity diagnostic predictions 
    using morphological data from cellular scans. 
    
    ### How it works:
    1. **Data Acquisition**: Metrics are extracted from digitized FNA images.
    2. **Processing**: Features are scaled using a standard normal distribution.
    3. **Inference**: The pre-trained Random Forest model evaluates the risk.
    4. **Visualization**: Real-time feedback is provided via interactive charts.
    
    *Note: This is a research prototype.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)
