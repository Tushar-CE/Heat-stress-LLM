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
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heat Stress AI",
    page_icon="🌡️",
    layout="wide"
)

# Clean, minimal CSS like DeepSeek
st.markdown("""
<style>
    .stApp {
        background: #0d1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .chat-message.user {
        background: #1c2333;
        border-left: 3px solid #58a6ff;
        color: #e6edf3;
    }
    .chat-message.assistant {
        background: #161b22;
        border-left: 3px solid #3fb950;
        color: #e6edf3;
    }
    .chat-message .role {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .chat-message.user .role {
        color: #58a6ff;
    }
    .chat-message.assistant .role {
        color: #3fb950;
    }
    .chat-message .content {
        white-space: pre-wrap;
    }
    .chat-message .content strong {
        color: #f0e6d0;
    }
    .chat-message .content ul, .chat-message .content ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .chat-message .content li {
        margin: 0.2rem 0;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0d1117;
        padding: 1rem 2rem;
        border-top: 1px solid #21262d;
        z-index: 100;
    }
    .input-container .stTextInput > div > div > input {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
    }
    .input-container .stTextInput > div > div > input:focus {
        border-color: #58a6ff !important;
        box-shadow: none !important;
    }
    .input-container .stTextInput > div > div > input::placeholder {
        color: #8b949e;
    }
    .input-container .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        transition: 0.2s !important;
        width: 100%;
    }
    .input-container .stButton > button:hover {
        background: #2ea043 !important;
        transform: scale(1.02);
    }
    .input-container .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .result-container {
        margin-bottom: 100px;
        padding: 0 1rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #161b22;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #21262d;
    }
    .metric-card .label {
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-card .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e6edf3;
        margin-top: 0.2rem;
    }
    .metric-card .value.high { color: #f85149; }
    .metric-card .value.moderate { color: #d29922; }
    .metric-card .value.low { color: #3fb950; }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge.critical { background: #da3633; color: white; }
    .status-badge.high { background: #d29922; color: white; }
    .status-badge.moderate { background: #d29922; color: white; }
    .status-badge.low { background: #238636; color: white; }
    .status-badge.comfortable { background: #238636; color: white; }
    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }
    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 0.7rem;
        padding: 1rem 0;
    }
    .stExpander {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    .stExpander > div {
        color: #e6edf3;
    }
    .stMarkdown {
        color: #e6edf3;
    }
    .stSidebar {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    .stSidebar .stMarkdown {
        color: #e6edf3;
    }
    .stNumberInput > div > div > input {
        background: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    .stSlider > div > div {
        color: #e6edf3;
    }
    .stSelectSlider > div {
        color: #e6edf3;
    }
    .stButton > button {
        background: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background: #30363d;
    }
    .suggestion-btn {
        background: #21262d !important;
        color: #8b949e !important;
        border: 1px solid #30363d !important;
        border-radius: 20px !important;
        padding: 0.3rem 1rem !important;
        font-size: 0.8rem !important;
        margin: 0.2rem !important;
        cursor: pointer !important;
        transition: 0.2s !important;
    }
    .suggestion-btn:hover {
        background: #30363d !important;
        color: #e6edf3 !important;
    }
    .plotly-container {
        background: #161b22;
        border-radius: 8px;
        border: 1px solid #21262d;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading and model training (same as before)
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
    "Rest (R)": {"M": 115, "PET_AL": 35, "description": "Sitting", "base_factor": 0.15},
    "Light (LW)": {"M": 180, "PET_AL": 35.5, "description": "Light hand work", "base_factor": 0.20},
    "Moderate (MW)": {"M": 300, "PET_AL": 32, "description": "Moderate lifting", "base_factor": 0.25},
    "Heavy (HW)": {"M": 415, "PET_AL": 31, "description": "Heavy lifting", "base_factor": 0.28},
    "Very Heavy (VHW)": {"M": 520, "PET_AL": 30, "description": "Very intense", "base_factor": 0.30}
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
        return "SEVERE DISCOMFORT"
    elif pmv >= 2.5:
        return "VERY UNCOMFORTABLE"
    elif pmv >= 2.0:
        return "MODERATE DISCOMFORT"
    elif pmv >= 1.5:
        return "MILD DISCOMFORT"
    elif pmv >= 1.0:
        return "SLIGHT WARMTH"
    else:
        return "COMFORTABLE"

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
                max_tokens=800
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
                "max_tokens": 800
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

def create_risk_chart(pet, pmv, ppd, productivity_loss):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("PET", "PMV", "PPD", "Productivity Loss"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=pet,
            title={'text': "°C"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [20, 50], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#f85149" if pet > 35 else "#d29922" if pet > 29 else "#3fb950"},
                'steps': [
                    {'range': [20, 23], 'color': "rgba(63,185,80,0.2)"},
                    {'range': [23, 29], 'color': "rgba(63,185,80,0.3)"},
                    {'range': [29, 35], 'color': "rgba(210,153,34,0.3)"},
                    {'range': [35, 41], 'color': "rgba(248,81,73,0.3)"},
                    {'range': [41, 50], 'color': "rgba(248,81,73,0.5)"}
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
            mode="gauge+number",
            value=pmv,
            title={'text': ""},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 3.5], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#f85149" if pmv > 2.5 else "#d29922" if pmv > 1.5 else "#3fb950"},
                'steps': [
                    {'range': [0, 1], 'color': "rgba(63,185,80,0.2)"},
                    {'range': [1, 1.5], 'color': "rgba(63,185,80,0.3)"},
                    {'range': [1.5, 2.5], 'color': "rgba(210,153,34,0.3)"},
                    {'range': [2.5, 3.5], 'color': "rgba(248,81,73,0.3)"}
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
            mode="gauge+number",
            value=ppd,
            title={'text': "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#f85149" if ppd > 50 else "#d29922" if ppd > 25 else "#3fb950"},
                'steps': [
                    {'range': [0, 25], 'color': "rgba(63,185,80,0.2)"},
                    {'range': [25, 50], 'color': "rgba(210,153,34,0.3)"},
                    {'range': [50, 100], 'color': "rgba(248,81,73,0.3)"}
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
            mode="gauge+number",
            value=productivity_loss,
            title={'text': "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 30], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#f85149" if productivity_loss > 20 else "#d29922" if productivity_loss > 10 else "#3fb950"},
                'steps': [
                    {'range': [0, 10], 'color': "rgba(63,185,80,0.2)"},
                    {'range': [10, 20], 'color': "rgba(210,153,34,0.3)"},
                    {'range': [20, 30], 'color': "rgba(248,81,73,0.3)"}
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
        height=450,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    
    return fig

def create_height_chart(height_profile):
    if not height_profile:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=height_profile['heights'],
        y=height_profile['pet_profile'],
        mode='lines',
        name='PET',
        line=dict(color='#3fb950', width=3),
        fill='tozeroy',
        fillcolor='rgba(63,185,80,0.1)'
    ))
    
    fig.add_hline(y=41, line_dash="dash", line_color="#f85149", annotation_text="Critical")
    fig.add_hline(y=35, line_dash="dash", line_color="#d29922", annotation_text="High")
    fig.add_hline(y=29, line_dash="dash", line_color="#d29922", annotation_text="Moderate")
    
    fig.update_layout(
        title="Vertical Temperature Profile",
        xaxis_title="Height (m)",
        yaxis_title="PET (°C)",
        template='plotly_dark',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x'
    )
    
    return fig

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'has_results' not in st.session_state:
    st.session_state.has_results = False
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Sidebar for inputs
with st.sidebar:
    st.markdown("### 🌡️ Input Data")
    st.markdown("---")
    
    T = st.number_input("Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    clo = st.select_slider("Clothing (clo)", options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00], value=0.57)
    met = st.select_slider("Activity (met)", options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0], value=3.2)
    height = st.slider("Working Height (m)", 0, 100, 0, 1)
    baseline_productivity = st.number_input("Baseline Output (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Setup")
    api_type = st.selectbox("Provider", ["openai", "deepseek"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
    if api_key:
        st.session_state.api_key = api_key
    
    if st.button("📊 Calculate Results", use_container_width=True):
        # Calculate results
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
        
        st.session_state.current_results = {
            'T': T, 'RH': RH, 'WS': WS, 'clo': clo, 'met': met, 'height': height,
            'ground_pet': ground_pet, 'ground_pmv': ground_pmv, 'ground_ppd': ground_ppd,
            'risk_level': risk_level, 'risk_desc': risk_desc, 'risk_icon': risk_icon,
            'risk_class': risk_class, 'pmv_interpretation': pmv_interpretation,
            'current_work_key': current_work_key, 'productivity_loss': productivity_loss,
            'height_profile': height_profile, 'baseline_productivity': baseline_productivity,
            'predictions': predictions
        }
        st.session_state.has_results = True
        st.rerun()

# Main content - Clean chat interface
st.markdown('<div class="main-header">🌡️ Heat Stress AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask about your heat stress results · Get instant analysis</div>', unsafe_allow_html=True)

# Display results if available
if st.session_state.has_results and st.session_state.current_results:
    r = st.session_state.current_results
    
    # Quick metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">🌡️ PET</div>
            <div class="value {'high' if r['ground_pet'] > 35 else 'moderate' if r['ground_pet'] > 29 else 'low'}">{r['ground_pet']:.1f}°C</div>
        </div>
        <div class="metric-card">
            <div class="label">📊 PMV</div>
            <div class="value {'high' if r['ground_pmv'] > 2.5 else 'moderate' if r['ground_pmv'] > 1.5 else 'low'}">{r['ground_pmv']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">😓 PPD</div>
            <div class="value {'high' if r['ground_ppd'] > 50 else 'moderate' if r['ground_ppd'] > 25 else 'low'}">{r['ground_ppd']:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">📉 Loss</div>
            <div class="value {'high' if r['productivity_loss'] > 20 else 'moderate' if r['productivity_loss'] > 10 else 'low'}">{r['productivity_loss']:.1f}%</div>
        </div>
    </div>
    <div style="text-align:center; margin:0.5rem 0;">
        <span class="status-badge {r['risk_class']}">{r['risk_icon']} {r['risk_level']}</span>
        <span style="color:#8b949e; margin-left:1rem;">{r['risk_desc']}</span>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_risk = create_risk_chart(r['ground_pet'], r['ground_pmv'], r['ground_ppd'], r['productivity_loss'])
        st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
    with col2:
        if r['height_profile']:
            fig_height = create_height_chart(r['height_profile'])
            if fig_height:
                st.plotly_chart(fig_height, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div style="background:#161b22; padding:0.8rem; border-radius:8px; border:1px solid #21262d; margin-top:0.5rem;">
            <div style="color:#8b949e; font-size:0.8rem;">Working Height</div>
            <div style="color:#e6edf3; font-size:1.2rem; font-weight:600;">{r['height']}m</div>
            <div style="color:#8b949e; font-size:0.8rem; margin-top:0.3rem;">Reduction: {r['height_profile']['reduction']:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

# Chat messages
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        role_label = "You" if msg["role"] == "user" else "AI Assistant"
        st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="role">{role_label}</div>
            <div class="content">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.messages and st.session_state.has_results:
        r = st.session_state.current_results
        welcome_msg = f"""**I've analyzed your site data. Here's what I found:**

• **Risk Level:** {r['risk_icon']} {r['risk_level']} - {r['risk_desc']}
• **PET:** {r['ground_pet']:.1f}°C (Threshold: 35°C for high risk)
• **PMV:** {r['ground_pmv']:.2f} - {r['pmv_interpretation']}
• **Productivity Impact:** {r['productivity_loss']:.1f}% loss for {r['current_work_key']}

**Ask me anything about your results:**"""
        
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="role">AI Assistant</div>
            <div class="content">{welcome_msg}</div>
        </div>
        """, unsafe_allow_html=True)

# Input area at bottom
with st.container():
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g., Why is my PET level so high?",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send_button = st.button("Send", use_container_width=True, key="send_btn")

# Suggestions
if not st.session_state.messages:
    suggestions = [
        "Why is my PET level so high?",
        "What should I do to reduce heat stress?",
        "How does humidity affect my risk?",
        "What's the best work schedule?",
        "Explain my productivity loss",
        "How does height affect temperature?"
    ]
    cols = st.columns(6)
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"sug_{i}"):
                user_input = suggestion
                st.rerun()

# Process user input
if send_button and user_input:
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar")
    elif not st.session_state.has_results:
        st.error("⚠️ Please calculate results first using the sidebar button")
    else:
        r = st.session_state.current_results
        
        context = f"""
        Site: {r['T']:.1f}°C, {r['RH']:.0f}% humidity, {r['WS']:.1f} m/s wind, {r['height']}m height
        Clothing: {r['clo']:.2f} clo, Activity: {r['met']:.1f} met ({r['current_work_key']})
        Results: PET {r['ground_pet']:.1f}°C ({r['risk_level']}), PMV {r['ground_pmv']:.2f}, PPD {r['ground_ppd']:.1f}%, Loss {r['productivity_loss']:.1f}%
        """
        
        system_message = """You are a construction heat stress safety expert. Give clear, practical answers about heat stress. Use bullet points. Be specific and actionable."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": f"Current Site Data: {context}"}
        ]
        
        for msg in st.session_state.messages[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🧠 Analyzing..."):
            response, error = call_llm_api(messages, api_key, api_type)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error}"})
        
        st.rerun()

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="footer">
    Heat Stress AI Assistant · Powered by AI · Enter your data in the sidebar
</div>
""", unsafe_allow_html=True)
