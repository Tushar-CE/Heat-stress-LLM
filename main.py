import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import hashlib
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heat Stress AI - Chat Assistant",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #0a0a1a 0%, #1a1a3e 50%, #0d1b2a 100%);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #4ECDC4, #44B39D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1.5rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #4ECDC4;
        border-radius: 10px;
    }
    .message {
        margin: 1rem 0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .message.user {
        background: rgba(78,205,196,0.1);
        border: 1px solid rgba(78,205,196,0.15);
        margin-left: 2rem;
        color: #ffffff;
    }
    .message.assistant {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        margin-right: 2rem;
        color: #e0e0e0;
    }
    .message .role {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .message.user .role {
        color: #4ECDC4;
    }
    .message.assistant .role {
        color: #FFD93D;
    }
    .message .content {
        line-height: 1.7;
        font-size: 0.95rem;
    }
    .message .content strong {
        color: #4ECDC4;
    }
    .message .content ul, .message .content ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .message .content li {
        margin: 0.3rem 0;
    }
    .input-area {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.5rem;
        margin-top: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        margin: 0.5rem 0;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        border-left: 3px solid #4ECDC4;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: 0.3s;
    }
    .metric-card:hover {
        background: rgba(255,255,255,0.08);
        transform: translateY(-2px);
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
    }
    .metric-card .value.high { color: #FF6B6B; }
    .metric-card .value.moderate { color: #FFD93D; }
    .metric-card .value.low { color: #4ECDC4; }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge.critical { background: #8B0000; color: white; }
    .status-badge.high { background: #DC3545; color: white; }
    .status-badge.moderate { background: #FF8C00; color: white; }
    .status-badge.low { background: #28A745; color: white; }
    .status-badge.comfortable { background: #1a7a3a; color: white; }
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .stTextInput > div > div > input {
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: none !important;
        border-color: transparent !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4, #44B39D);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(78,205,196,0.3);
    }
    .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    [data-testid="stSidebar"] {
        background: rgba(10,10,30,0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #ffffff;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }
    .stSlider > div > div {
        color: #ffffff;
    }
    .stSelectSlider > div {
        color: #ffffff;
    }
    .stExpander {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stExpander > div {
        color: #ffffff;
    }
    .suggestion-chip {
        display: inline-block;
        background: rgba(78,205,196,0.1);
        border: 1px solid rgba(78,205,196,0.15);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        margin: 0.2rem;
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        cursor: pointer;
        transition: 0.3s;
    }
    .suggestion-chip:hover {
        background: rgba(78,205,196,0.2);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    np.random.seed(42)
    n_samples = 500
    T = np.random.uniform(20, 45, n_samples)
    RH = np.random.uniform(30, 90, n_samples)
    WS = np.random.uniform(0.1, 8, n_samples)
    PET = T + 5 - 0.8 * WS + 0.015 * (RH - 40) + np.random.normal(0, 1, n_samples)
    PET = np.clip(PET, 20, 50)
    PMV = 1.5 + (T - 25) * 0.12 - 0.15 * WS + 0.02 * (RH - 40) + np.random.normal(0, 0.2, n_samples)
    PMV = np.clip(PMV, 0, 4)
    df = pd.DataFrame({
        'T': T, 'RH': RH, 'WS': WS, 'PET': PET, 'PMV': PMV,
        'PPD': 5 + 85 * (1 - np.exp(-0.5 * (PMV - 1))),
        'SET': PET - 1 + np.random.normal(0, 0.5, n_samples),
        'RWS': WS * 0.8 + np.random.normal(0, 0.2, n_samples),
        'CE': 2 + 0.5 * WS + np.random.normal(0, 0.3, n_samples),
        'Height': np.random.uniform(0, 100, n_samples),
        'PETH': PET - np.random.uniform(0, 6, n_samples),
        'PMVH': PMV - np.random.uniform(0, 1, n_samples)
    })
    return df

CSV_PATH = "EXBD.csv"

if not os.path.exists(CSV_PATH):
    df = create_sample_data()
    df.to_csv(CSV_PATH, index=False)
else:
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except:
        df = create_sample_data()

hs_df = pd.DataFrame({
    'T(0C)': pd.to_numeric(df['T'], errors='coerce'),
    'RH(%)': pd.to_numeric(df['RH'], errors='coerce'),
    'WS(m/s)': pd.to_numeric(df['WS'], errors='coerce'),
    'PET(0C)': pd.to_numeric(df['PET'], errors='coerce'),
    'PMV': pd.to_numeric(df['PMV'], errors='coerce'),
    'PPD(%)': pd.to_numeric(df['PPD'], errors='coerce'),
    'SET (0C)': pd.to_numeric(df['SET'], errors='coerce'),
    'RWS(m/s)': pd.to_numeric(df['RWS'], errors='coerce'),
    'CE(0C)': pd.to_numeric(df['CE'], errors='coerce')
}).dropna()

bh_df = pd.DataFrame({
    'Height(m)': pd.to_numeric(df['Height'], errors='coerce'),
    'PET(0C)': pd.to_numeric(df['PETH'], errors='coerce'),
    'PMV': pd.to_numeric(df['PMVH'], errors='coerce')
}).dropna()

if len(bh_df) > 0:
    bh_df = bh_df[bh_df['Height(m)'] > 0].sort_values('Height(m)')

features = ['T(0C)', 'RH(%)', 'WS(m/s)']
targets = ['PET(0C)', 'PMV', 'PPD(%)', 'SET (0C)', 'RWS(m/s)', 'CE(0C)']

X = hs_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_dict = {}
for target in targets:
    if target in hs_df.columns:
        y_dict[target] = hs_df[target].values

models = {}
for target, y in y_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    nn = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    nn.fit(X_train, y_train)
    models[target] = nn

work_data = {
    "Rest (R)": {"M": 115, "PET_AL": 35, "description": "Sitting, light activities", "base_factor": 0.15},
    "Light (LW)": {"M": 180, "PET_AL": 35.5, "description": "Standing, light hand work", "base_factor": 0.20},
    "Moderate (MW)": {"M": 300, "PET_AL": 32, "description": "Walking, moderate lifting", "base_factor": 0.25},
    "Heavy (HW)": {"M": 415, "PET_AL": 31, "description": "Heavy lifting, shoveling", "base_factor": 0.28},
    "Very Heavy (VHW)": {"M": 520, "PET_AL": 30, "description": "Very intense labor", "base_factor": 0.30}
}

activity_to_work = {
    2.1: "Light (LW)", 2.2: "Light (LW)", 2.6: "Moderate (MW)",
    3.2: "Heavy (HW)", 3.8: "Heavy (HW)", 4.0: "Very Heavy (VHW)"
}

def calc_productivity_loss(pet_value, work_type_key, baseline):
    work = work_data[work_type_key]
    PET_AL = work["PET_AL"]
    base_factor = work["base_factor"]
    if pet_value <= PET_AL:
        return 0
    else:
        delta_pet = pet_value - PET_AL
        PL = 30 * (1 - np.exp(-base_factor * delta_pet))
        return min(PL, 30)

def get_thermal_risk_level(pet):
    if pet > 41.0:
        return "CRITICAL", "Immediate work stoppage required", "🔴", "critical"
    elif pet > 35.0:
        return "HIGH", "Severe heat strain - reduced work capacity", "🟠", "high"
    elif pet > 29.0:
        return "MODERATE", "Elevated heat stress - monitoring required", "🟡", "moderate"
    elif pet > 23.0:
        return "LOW", "Mild heat stress - standard precautions", "🟢", "low"
    else:
        return "COMFORTABLE", "Optimal working conditions", "✅", "comfortable"

def get_pmv_interpretation(pmv):
    if pmv >= 3.0:
        return "SEVERE DISCOMFORT - High heat strain"
    elif pmv >= 2.5:
        return "VERY UNCOMFORTABLE - Significant heat stress"
    elif pmv >= 2.0:
        return "MODERATE DISCOMFORT - Noticeable heat"
    elif pmv >= 1.5:
        return "MILD DISCOMFORT - Warm conditions"
    elif pmv >= 1.0:
        return "SLIGHT WARMTH - Acceptable"
    else:
        return "COMFORTABLE - Neutral thermal sensation"

def get_height_profile(ground_pet, ground_pmv, height_m, bh_df):
    if len(bh_df) == 0:
        return None
    heights_original = bh_df['Height(m)'].values
    pet_pattern = bh_df['PET(0C)'].values.copy()
    pmv_pattern = bh_df['PMV'].values.copy() if 'PMV' in bh_df.columns else pet_pattern * 0.08
    for i in range(1, len(pet_pattern)):
        if pet_pattern[i] > pet_pattern[i-1]:
            pet_pattern[i] = max(pet_pattern[i-1] - 0.1, 0)
    for i in range(1, len(pmv_pattern)):
        if pmv_pattern[i] > pmv_pattern[i-1]:
            pmv_pattern[i] = max(pmv_pattern[i-1] - 0.01, -3)
    if pet_pattern[0] != 0:
        pet_relative = pet_pattern / pet_pattern[0]
    else:
        pet_relative = pet_pattern
    if pmv_pattern[0] != 0:
        pmv_relative = pmv_pattern / pmv_pattern[0]
    else:
        pmv_relative = pmv_pattern
    pet_profile = ground_pet * pet_relative
    pmv_profile = ground_pmv * pmv_relative
    heights_smooth = np.linspace(0, 100, 100)
    pet_smooth = np.interp(heights_smooth, heights_original, pet_profile)
    pmv_smooth = np.interp(heights_smooth, heights_original, pmv_profile)
    if 0 <= height_m <= 100:
        pet_at_height = np.interp(height_m, heights_smooth, pet_smooth)
        pmv_at_height = np.interp(height_m, heights_smooth, pmv_smooth)
    else:
        pet_at_height = ground_pet
        pmv_at_height = ground_pmv
    pet_at_100 = np.interp(100, heights_smooth, pet_smooth)
    pmv_at_100 = np.interp(100, heights_smooth, pmv_smooth)
    return {
        'pet_at_height': pet_at_height,
        'pmv_at_height': pmv_at_height,
        'pet_at_100': pet_at_100,
        'pmv_at_100': pmv_at_100,
        'pet_profile': pet_smooth,
        'pmv_profile': pmv_smooth,
        'heights': heights_smooth,
        'reduction': ground_pet - pet_at_height,
        'lapse_rate': (ground_pet - pet_at_100) / 100 if height_m > 0 else 0
    }

def call_llm_api(messages, api_key, api_type="openai"):
    if not api_key:
        return None, "Please enter your API key in the sidebar"
    try:
        if api_type == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content, None
        elif api_type == "deepseek":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'], None
            else:
                return None, f"API Error: {response.status_code}"
        else:
            return None, "Unsupported API type"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_risk_chart(pet, pmv, ppd, productivity_loss, height_profile):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("PET Risk Level", "PMV Comfort Scale", "PPD Distribution", "Productivity Impact"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=pet,
            title={'text': "PET (°C)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [20, 50], 'tickwidth': 1},
                'bar': {'color': "#FF6B6B" if pet > 35 else "#FFD93D" if pet > 29 else "#4ECDC4"},
                'steps': [
                    {'range': [20, 23], 'color': "rgba(78,205,196,0.2)"},
                    {'range': [23, 29], 'color': "rgba(78,205,196,0.3)"},
                    {'range': [29, 35], 'color': "rgba(255,217,61,0.3)"},
                    {'range': [35, 41], 'color': "rgba(255,107,107,0.3)"},
                    {'range': [41, 50], 'color': "rgba(139,0,0,0.4)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': pet
                }
            }
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=pmv,
            title={'text': "PMV"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 3.5], 'tickwidth': 1},
                'bar': {'color': "#FF6B6B" if pmv > 2.5 else "#FFD93D" if pmv > 1.5 else "#4ECDC4"},
                'steps': [
                    {'range': [0, 1], 'color': "rgba(78,205,196,0.2)"},
                    {'range': [1, 1.5], 'color': "rgba(78,205,196,0.3)"},
                    {'range': [1.5, 2.5], 'color': "rgba(255,217,61,0.3)"},
                    {'range': [2.5, 3.5], 'color': "rgba(255,107,107,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': pmv
                }
            }
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=ppd,
            title={'text': "PPD (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#FF6B6B" if ppd > 50 else "#FFD93D" if ppd > 25 else "#4ECDC4"},
                'steps': [
                    {'range': [0, 25], 'color': "rgba(78,205,196,0.2)"},
                    {'range': [25, 50], 'color': "rgba(255,217,61,0.3)"},
                    {'range': [50, 100], 'color': "rgba(255,107,107,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': ppd
                }
            }
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=productivity_loss,
            title={'text': "Productivity Loss (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 30], 'tickwidth': 1},
                'bar': {'color': "#FF6B6B" if productivity_loss > 20 else "#FFD93D" if productivity_loss > 10 else "#4ECDC4"},
                'steps': [
                    {'range': [0, 10], 'color': "rgba(78,205,196,0.2)"},
                    {'range': [10, 20], 'color': "rgba(255,217,61,0.3)"},
                    {'range': [20, 30], 'color': "rgba(255,107,107,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': productivity_loss
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_height_chart(height_profile, ground_pet):
    if not height_profile:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=height_profile['heights'],
        y=height_profile['pet_profile'],
        mode='lines',
        name='PET Profile',
        line=dict(color='#4ECDC4', width=3),
        fill='tozeroy',
        fillcolor='rgba(78,205,196,0.1)'
    ))
    
    fig.add_hline(y=41, line_dash="dash", line_color="#FF6B6B", annotation_text="Critical (>41°C)")
    fig.add_hline(y=35, line_dash="dash", line_color="#FF8C00", annotation_text="High (>35°C)")
    fig.add_hline(y=29, line_dash="dash", line_color="#FFD93D", annotation_text="Moderate (>29°C)")
    
    fig.update_layout(
        title="Vertical PET Profile",
        xaxis_title="Height Above Ground (m)",
        yaxis_title="PET (°C)",
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x'
    )
    
    return fig

st.markdown('<div class="main-title">🧬 Heat Stress AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turn your site data into actionable heat stress insights with AI-powered analysis</div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'first_message' not in st.session_state:
    st.session_state.first_message = True

with st.sidebar:
    st.markdown("### ⚙️ Site Conditions")
    st.markdown("---")
    
    st.markdown("#### 🌤️ Environmental")
    T = st.number_input("Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    
    st.markdown("---")
    st.markdown("#### 👷 Personal")
    clo = st.select_slider("Clothing (clo)", options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00], value=0.57)
    met = st.select_slider("Activity (met)", options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0], value=3.2)
    height = st.slider("Working Height (m)", 0, 100, 0, 1)
    
    st.markdown("---")
    st.markdown("#### 📊 Productivity")
    baseline_productivity = st.number_input("Baseline Output (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Configuration")
    api_type = st.selectbox("AI Provider", ["openai", "deepseek"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
    if api_key:
        st.session_state.api_key = api_key
    
    if st.button("🔄 Update Results", use_container_width=True):
        st.rerun()

input_data = np.array([[T, RH, WS]])
input_scaled = scaler.transform(input_data)

predictions = {}
for target in targets:
    if target in models:
        predictions[target] = models[target].predict(input_scaled)[0]
    else:
        if target == 'PET(0C)':
            predictions[target] = T + 5 + 0.015*(RH-40) - 0.8*WS
        elif target == 'PMV':
            predictions[target] = 1.5 + (T-25)*0.12 - 0.15*WS + 0.02*(RH-40)
        elif target == 'PPD(%)':
            predictions[target] = 50
        elif target == 'SET (0C)':
            predictions[target] = T + 3
        elif target == 'RWS(m/s)':
            predictions[target] = WS * 0.8
        elif target == 'CE(0C)':
            predictions[target] = 2 + 0.5*WS
        else:
            predictions[target] = 0

predictions['PET(0C)'] = np.clip(predictions['PET(0C)'] + clo * 0.5 + (met - 2.0) * 0.3, 20, 50)
predictions['PMV'] = np.clip(predictions['PMV'] + clo * 0.3 + (met - 2.0) * 0.2, 0, 3.5)
predictions['PPD(%)'] = np.clip(predictions['PPD(%)'] + clo * 2 + (met - 2.0) * 1.5, 5, 90)

ground_pet = predictions["PET(0C)"]
ground_pmv = predictions["PMV"]
ground_ppd = predictions['PPD(%)']

height_profile = get_height_profile(ground_pet, ground_pmv, height, bh_df)
risk_level, risk_desc, risk_icon, risk_class = get_thermal_risk_level(ground_pet)
pmv_interpretation = get_pmv_interpretation(ground_pmv)
current_work_key = activity_to_work.get(met, "Heavy (HW)")
pet_effective = height_profile['pet_at_height'] if height_profile else ground_pet
productivity_loss = calc_productivity_loss(pet_effective, current_work_key, baseline_productivity)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">🌡️ PET</div>
        <div class="value {'high' if ground_pet > 35 else 'moderate' if ground_pet > 29 else 'low'}">{ground_pet:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">📊 PMV</div>
        <div class="value {'high' if ground_pmv > 2.5 else 'moderate' if ground_pmv > 1.5 else 'low'}">{ground_pmv:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">😓 PPD</div>
        <div class="value {'high' if ground_ppd > 50 else 'moderate' if ground_ppd > 25 else 'low'}">{ground_ppd:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">📉 Loss</div>
        <div class="value {'high' if productivity_loss > 20 else 'moderate' if productivity_loss > 10 else 'low'}">{productivity_loss:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; margin:0.5rem 0;">
    <span class="status-badge {risk_class}">{risk_icon} {risk_level} - {risk_desc}</span>
    <span style="color:rgba(255,255,255,0.4); margin-left:1rem;">|</span>
    <span style="color:rgba(255,255,255,0.6); margin-left:1rem;">{pmv_interpretation}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div style="font-size:1.2rem; font-weight:600; color:#4ECDC4; margin:0.5rem 0;">💬 Ask anything about your heat stress results</div>', unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    chat_html = f'<div class="chat-container">'
    
    if st.session_state.first_message:
        chat_html += f'''
        <div class="message assistant">
            <div class="role">🧠 AI Assistant</div>
            <div class="content">
                <strong>Welcome to Heat Stress AI Assistant!</strong><br><br>
                I can help you understand your heat stress results and provide guidance. Here's what I see:
                <br><br>
                • <strong>Current Risk Level:</strong> {risk_icon} {risk_level} - {risk_desc}
                <br>
                • <strong>PET:</strong> {ground_pet:.1f}°C (Threshold: 35°C for high risk)
                <br>
                • <strong>PMV:</strong> {ground_pmv:.2f} - {pmv_interpretation}
                <br>
                • <strong>Productivity Impact:</strong> {productivity_loss:.1f}% loss for {current_work_key}
                <br><br>
                <strong>Try asking:</strong>
                <br>
                • "Why is my PET level so high?"
                <br>
                • "What should I do to reduce heat stress?"
                <br>
                • "How does humidity affect my risk?"
                <br>
                • "What's the best work schedule for these conditions?"
            </div>
        </div>
        '''
    
    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "assistant"
        role_label = "👤 You" if msg["role"] == "user" else "🧠 AI Assistant"
        chat_html += f'''
        <div class="message {role_class}">
            <div class="role">{role_label}</div>
            <div class="content">{msg["content"]}</div>
        </div>
        '''
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input(
        "Ask a question",
        placeholder="e.g., Why is my PET level so high?",
        label_visibility="collapsed",
        key="user_input"
    )
with col2:
    ask_button = st.button("Send", use_container_width=True)

suggestions = [
    "Why is my PET level so high?",
    "What should I do to reduce heat stress?",
    "How does humidity affect my risk?",
    "What's the best work schedule?",
    "Explain my productivity loss",
    "How does height affect temperature?"
]

st.markdown('<div style="margin:0.5rem 0; display:flex; flex-wrap:wrap; gap:0.3rem;">', unsafe_allow_html=True)
for suggestion in suggestions:
    if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=False):
        user_question = suggestion
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if ask_button and user_question:
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar to use the AI assistant")
    else:
        context = f"""
        Site Conditions:
        - Temperature: {T:.1f}°C
        - Humidity: {RH:.0f}%
        - Wind Speed: {WS:.1f} m/s
        - Working Height: {height}m
        - Clothing: {clo:.2f} clo
        - Activity: {met:.1f} met ({current_work_key})
        
        Results:
        - PET: {ground_pet:.1f}°C ({risk_level} risk)
        - PMV: {ground_pmv:.2f} ({pmv_interpretation})
        - PPD: {ground_ppd:.1f}%
        - Productivity Loss: {productivity_loss:.1f}% for {current_work_key}
        """
        
        system_message = """You are a construction heat stress safety expert. Provide clear, professional, and actionable answers. Use the provided site data to give specific recommendations. Include practical guidance for site management and workers. Be concise but thorough."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": f"Current Site Data: {context}"}
        ]
        
        for msg in st.session_state.chat_history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_question})
        
        with st.spinner("🧠 Analyzing with AI..."):
            response, error = call_llm_api(messages, api_key, api_type)
        
        if response:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.first_message = False
            st.rerun()
        else:
            st.error(f"Error: {error}")

st.markdown("---")

st.markdown('<div class="section-title">📊 Risk Visualization</div>', unsafe_allow_html=True)

fig_risk = create_risk_chart(ground_pet, ground_pmv, ground_ppd, productivity_loss, height_profile)
st.plotly_chart(fig_risk, use_container_width=True)

if height_profile:
    fig_height = create_height_chart(height_profile, ground_pet)
    if fig_height:
        st.plotly_chart(fig_height, use_container_width=True)

with st.expander("📐 Understanding Your Results"):
    st.markdown("""
    ### How to Interpret Your Results
    
    **PET (Physiological Equivalent Temperature)**
    - The temperature at which your body would feel the same thermal stress
    - Higher PET = more heat stress on your body
    
    **Risk Levels:**
    - **Critical (>41°C)**: Emergency - Stop all work immediately
    - **High (35-41°C)**: Severe strain - 45 min work / 15 min rest
    - **Moderate (29-35°C)**: Elevated stress - 60 min work / 10 min rest
    - **Low (23-29°C)**: Mild stress - Standard precautions
    - **Comfortable (<23°C)**: Optimal conditions - Normal operations
    
    **PMV (Predicted Mean Vote)**
    - Measures thermal comfort on a scale of 0 (neutral) to 3.5 (very hot)
    - Higher values indicate greater discomfort
    
    **PPD (Predicted Percentage Dissatisfied)**
    - Percentage of workers expected to feel uncomfortable
    - Higher percentages indicate need for intervention
    
    **Productivity Loss**
    - Estimated reduction in work output due to heat stress
    - Based on construction site empirical studies
    
    **Height Impact**
    - PET decreases by 0.05-0.08°C per meter of elevation
    - Working at higher levels provides natural cooling relief
    """)

st.markdown("""
<div class="footer">
    🧬 Heat Stress AI Assistant | Powered by AI | Version 1.0
</div>
""", unsafe_allow_html=True)
