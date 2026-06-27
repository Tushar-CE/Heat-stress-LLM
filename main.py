import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import json
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Construction Heat Stress - AI Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        padding: 1.5rem;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        padding: 0.6rem 1rem;
        margin: 1.2rem 0 0.8rem 0;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        border-left: 5px solid #4ECDC4;
    }
    .result-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(8px);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0.4rem 0;
        transition: 0.3s;
    }
    .result-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-2px);
    }
    .result-label {
        font-weight: 500;
        color: rgba(255,255,255,0.8);
        font-size: 0.85rem;
    }
    .result-value {
        font-weight: 700;
        color: #ffffff;
        font-size: 1.2rem;
    }
    .result-value.high {
        color: #FF6B6B;
    }
    .result-value.moderate {
        color: #FFD93D;
    }
    .result-value.low {
        color: #4ECDC4;
    }
    .status-box {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: #ffffff;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .status-box.critical {
        background: linear-gradient(135deg, #8B0000, #FF0000);
    }
    .status-box.high {
        background: linear-gradient(135deg, #FF4500, #FF8C00);
    }
    .status-box.moderate {
        background: linear-gradient(135deg, #FFA500, #FFD700);
        color: #333;
    }
    .status-box.low {
        background: linear-gradient(135deg, #006400, #228B22);
    }
    .status-box.comfortable {
        background: linear-gradient(135deg, #1a472a, #2d7d46);
    }
    .explanation-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ffffff;
        margin: 0.8rem 0;
        line-height: 1.8;
    }
    .explanation-box h4 {
        color: #4ECDC4;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .explanation-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .explanation-box li {
        margin: 0.3rem 0;
    }
    .factor-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .factor-card .factor-name {
        font-weight: 600;
        color: #4ECDC4;
        font-size: 1.05rem;
    }
    .factor-card .factor-impact {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        color: rgba(255,255,255,0.8);
    }
    .factor-card .factor-detail {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.2rem;
    }
    .metric-item {
        background: rgba(255,255,255,0.06);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-item .label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-item .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
    }
    .guidance-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #4ECDC4;
        color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    .guidance-box.critical {
        border-left-color: #FF0000;
        background: rgba(255,0,0,0.08);
    }
    .guidance-box.high {
        border-left-color: #FF8C00;
        background: rgba(255,140,0,0.08);
    }
    .guidance-box.moderate {
        border-left-color: #FFD700;
        background: rgba(255,215,0,0.08);
    }
    .guidance-box.low {
        border-left-color: #4ECDC4;
        background: rgba(78,205,196,0.08);
    }
    .llm-response {
        background: rgba(78,205,196,0.05);
        border: 1px solid rgba(78,205,196,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #ffffff;
        line-height: 1.8;
    }
    .llm-response h3 {
        color: #4ECDC4;
        margin-top: 0;
    }
    .llm-response strong {
        color: #FFD93D;
    }
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(195deg, #1a2f3f 0%, #1e3a4a 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #ffffff;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stSlider > div > div {
        color: #ffffff;
    }
    .stSelectSlider > div {
        color: #ffffff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1a3a47, #2a4a5a, #3a5a6a);
        transform: scale(1.02);
    }
    .stExpander {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stExpander > div {
        color: #ffffff;
    }
    .stSpinner > div {
        border-color: #4ECDC4 !important;
    }
    .stAlert {
        background: rgba(255,255,255,0.05) !important;
        color: #ffffff !important;
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

def generate_llm_explanation(T, RH, WS, clo, met, height, ground_pet, ground_pmv, risk_level, risk_desc, pmv_interpretation, productivity_loss, current_work_key, height_profile):
    """Generate comprehensive explanation using LLM"""
    
    # Prepare context data
    pet_at_height = height_profile['pet_at_height'] if height_profile else ground_pet
    reduction = height_profile['reduction'] if height_profile else 0
    
    # Determine primary contributing factors
    factors = []
    if T > 35:
        factors.append(f"High ambient temperature ({T:.1f}°C) is the primary driver of heat stress")
    elif T > 30:
        factors.append(f"Elevated temperature ({T:.1f}°C) contributes significantly to thermal load")
    else:
        factors.append(f"Moderate temperature ({T:.1f}°C) is manageable")
    
    if RH > 70:
        factors.append(f"High humidity ({RH:.0f}%) reduces evaporative cooling effectiveness")
    elif RH > 60:
        factors.append(f"Moderate humidity ({RH:.0f}%) somewhat reduces sweat evaporation")
    
    if WS < 1.0:
        factors.append(f"Low wind speed ({WS:.1f} m/s) limits convective heat dissipation")
    elif WS > 3.0:
        factors.append(f"Good air movement ({WS:.1f} m/s) enhances cooling")
    
    if clo > 0.8:
        factors.append(f"Heavy clothing ({clo:.2f} clo) traps body heat")
    elif clo < 0.5:
        factors.append(f"Light clothing ({clo:.2f} clo) allows better heat dissipation")
    
    if met > 3.5:
        factors.append(f"High metabolic activity ({met:.1f} met) generates significant internal heat")
    elif met > 2.8:
        factors.append(f"Moderate work intensity ({met:.1f} met) produces substantial metabolic heat")
    
    if height > 20:
        factors.append(f"Working at {height}m elevation provides {reduction:.1f}°C thermal relief")
    elif height > 0:
        factors.append(f"Working at {height}m provides minimal thermal benefit")
    
    if productivity_loss > 20:
        factors.append(f"Severe productivity loss ({productivity_loss:.1f}%) indicates critical heat strain")
    elif productivity_loss > 10:
        factors.append(f"Noticeable productivity impact ({productivity_loss:.1f}%) from heat exposure")
    
    # Prepare for LLM API call
    system_prompt = """You are a construction safety expert specializing in heat stress management. 
    Provide clear, professional explanations about heat stress risk factors and actionable guidance.
    Use bullet points for readability. Be specific and practical."""
    
    user_prompt = f"""Analyze this construction site heat stress scenario and provide a comprehensive explanation:

**Site Conditions:**
- Temperature: {T:.1f}°C
- Humidity: {RH:.0f}%
- Wind Speed: {WS:.1f} m/s
- Working Height: {height}m
- Clothing: {clo:.2f} clo
- Activity: {met:.1f} met ({current_work_key})

**Results:**
- PET: {ground_pet:.1f}°C ({risk_level} risk - {risk_desc})
- PMV: {ground_pmv:.2f} ({pmv_interpretation})
- Productivity Loss: {productivity_loss:.1f}%

**Key Factors Identified:**
{chr(10).join(['- ' + f for f in factors])}

Please provide:

1. **WHY this risk level exists** - Explain the combination of factors causing this heat stress level
2. **WHAT factors matter most** - Rank and explain the top 3 most critical factors
3. **HOW to address each factor** - Specific, actionable recommendations for each factor
4. **PRACTICAL guidance** for site management and workers

Keep the explanation professional, specific to construction, and actionable. Use bullet points."""
    
    return system_prompt, user_prompt, factors

def call_llm_api(system_prompt, user_prompt, api_key=None, api_type="openai"):
    """Call LLM API with error handling"""
    
    # Check for API key in session state or environment
    if api_key is None:
        api_key = st.session_state.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
    
    if not api_key:
        return None, "Please enter your API key in the sidebar"
    
    try:
        if api_type == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
                return None, f"API Error: {response.status_code} - {response.text}"
        
        else:
            return None, "Unsupported API type"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

st.markdown('<div class="main-title">🧠 Construction Heat Stress - AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get detailed explanations and guidance from AI about your heat stress risk</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Site Conditions")
    st.markdown("---")
    
    st.markdown("#### 🌤️ Environmental Parameters")
    T = st.number_input("Air Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("Relative Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    
    st.markdown("---")
    st.markdown("#### 👷 Worker Factors")
    clo = st.select_slider(
        "Clothing Insulation (clo)",
        options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00],
        value=0.57
    )
    met = st.select_slider(
        "Metabolic Rate (met)",
        options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0],
        value=3.2
    )
    
    st.markdown("---")
    st.markdown("#### 🏗️ Work Location")
    height = st.slider("Working Height Above Ground (m)", 0, 100, 0, 1)
    st.caption("PET decreases ~0.05-0.08°C per meter elevation")
    
    st.markdown("---")
    st.markdown("#### 📊 Productivity Baseline")
    baseline_productivity = st.number_input("Baseline Output (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Configuration")
    api_type = st.selectbox("API Type", ["openai", "deepseek"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
    if api_key:
        st.session_state.api_key = api_key
    
    use_llm = st.checkbox("Enable AI Explanations", value=True)
    
    st.markdown("---")
    if st.button("🔄 Analyze with AI", use_container_width=True):
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

height_profile = get_height_profile(ground_pet, ground_pmv, height, bh_df)

risk_level, risk_desc, risk_icon, risk_class = get_thermal_risk_level(ground_pet)
pmv_interpretation = get_pmv_interpretation(ground_pmv)

current_work_key = activity_to_work.get(met, "Heavy (HW)")
pet_effective = height_profile['pet_at_height'] if height_profile else ground_pet
productivity_loss = calc_productivity_loss(pet_effective, current_work_key, baseline_productivity)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="status-box {risk_class}">
        <div style="font-size: 2.5rem; font-weight: 700;">
            {risk_icon} {risk_level} RISK
        </div>
        <div style="font-size: 1.2rem; margin-top: 0.3rem;">
            {risk_desc}
        </div>
        <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">
            PET = {ground_pet:.1f}°C | PMV = {ground_pmv:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:12px; text-align:center; border:1px solid rgba(255,255,255,0.1); height:100%; display:flex; flex-direction:column; justify-content:center;">
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6);">Productivity Impact</div>
        <div style="font-size:2rem; font-weight:700; color:{'#FF6B6B' if productivity_loss > 20 else '#FFD93D' if productivity_loss > 10 else '#4ECDC4'};">{productivity_loss:.1f}%</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.5);">{current_work_key}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-title">📊 Thermal Exposure Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">🌡️ PET</div>
        <div class="value">{ground_pet:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">📊 PMV</div>
        <div class="value">{ground_pmv:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">😓 PPD</div>
        <div class="value">{predictions['PPD(%)']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">🌬️ Wind</div>
        <div class="value">{WS:.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

if height_profile:
    st.markdown('<div class="section-title">🏗️ Height Impact</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Ground Level (0m)</div>
            <div class="result-value">{ground_pet:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="result-card" style="border-color:#4ECDC4;">
            <div class="result-label">Working Height ({height}m)</div>
            <div class="result-value" style="color:#4ECDC4;">{height_profile['pet_at_height']:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Reduction</div>
            <div class="result-value" style="color:#FFD93D;">{height_profile['reduction']:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

if use_llm:
    st.markdown('<div class="section-title">🧠 AI Explanation & Guidance</div>', unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to use AI explanations")
    else:
        with st.spinner("🧠 Generating AI explanation..."):
            system_prompt, user_prompt, factors = generate_llm_explanation(
                T, RH, WS, clo, met, height, ground_pet, ground_pmv, 
                risk_level, risk_desc, pmv_interpretation, productivity_loss, 
                current_work_key, height_profile
            )
            
            explanation, error = call_llm_api(system_prompt, user_prompt, api_key, api_type)
            
            if explanation:
                st.markdown(f"""
                <div class="llm-response">
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 View Key Factors Identified"):
                    for i, f in enumerate(factors, 1):
                        st.markdown(f"**{i}.** {f}")
            else:
                st.error(f"AI Error: {error}")
                st.info("💡 You can still use the standard guidance below")

st.markdown('<div class="section-title">🛡️ Standard Safety Guidance</div>', unsafe_allow_html=True)

risk_level_effective, risk_desc_effective, _, risk_class_effective = get_thermal_risk_level(pet_effective)

if risk_class_effective == "critical":
    guidance = """
    🔴 **EMERGENCY PROTOCOL — IMMEDIATE ACTION REQUIRED**
    
    • **Work Stoppage**: All non-essential outdoor construction activities must cease immediately
    • **Worker Relocation**: Move all personnel to air-conditioned rest areas or shaded locations
    • **Hydration Protocol**: Mandatory 250ml water intake every 15 minutes with electrolyte supplementation
    • **Medical Monitoring**: Activate site heat stress response team; monitor for heat exhaustion symptoms
    • **High-Risk Workers**: Prioritize medical evaluation for workers with pre-existing conditions
    • **Resumption Criteria**: Only resume work when PET drops below 35°C with adequate cooling measures
    """
elif risk_class_effective == "high":
    guidance = f"""
    🟠 **HIGH HEAT STRESS — ENHANCED PRECAUTIONS**
    
    • **Work-Rest Cycles**: Implement 45-minute work / 15-minute rest rotation in shaded areas
    • **Cooling Measures**: Provide cooling vests, ice packs, and misting stations at work locations
    • **Hydration**: 250ml water every 30 minutes; electrolyte drinks recommended for heavy work
    • **Monitoring**: Designate safety officer to monitor worker condition and PET levels hourly
    • **Work Modifications**: Schedule heavy work (VHW, HW) during cooler morning/evening hours
    • **Height Consideration**: Utilize {height}m elevation for thermal relief
    """
elif risk_class_effective == "moderate":
    guidance = f"""
    🟡 **MODERATE HEAT STRESS — STANDARD PRECAUTIONS**
    
    • **Work-Rest Cycles**: 60-minute work / 10-minute rest in shaded areas for moderate activities
    • **Hydration**: 250ml water every 45 minutes; standard electrolyte availability
    • **Monitoring**: Regular observation of workers for early signs of heat strain
    • **Work Scheduling**: Continue normal operations with increased vigilance
    • **Cooling**: Ensure adequate shade and ventilation at all workstations
    • **Height Strategy**: Working at {height}m provides thermal reduction — consider vertical work positioning
    """
else:
    guidance = f"""
    🟢 **NORMAL OPERATIONS — ROUTINE MONITORING**
    
    • **Work Schedule**: Standard work operations with regular breaks
    • **Hydration**: Maintain normal water intake (250ml per hour minimum)
    • **Monitoring**: Continue routine safety observations
    • **Preparation**: Maintain readiness for temperature increases during peak hours
    • **Best Practice**: Working at {height}m provides optimal thermal conditions
    """

st.markdown(f"""
<div class="guidance-box {risk_class_effective}">
    {guidance}
</div>
""", unsafe_allow_html=True)

with st.expander("📐 Technical Notes & Methodology"):
    st.markdown("""
    **Thermal Exposure Assessment Parameters**
    
    | Parameter | Description | Application |
    |-----------|-------------|-------------|
    | **PET** | Physiological Equivalent Temperature (°C) | Primary heat stress indicator representing thermal sensation |
    | **PMV** | Predicted Mean Vote (0-3.5 scale) | Thermal comfort assessment for construction workers |
    | **PPD** | Predicted Percentage Dissatisfied (%) | Workforce acceptance of thermal conditions |
    
    **Heat Stress Risk Thresholds**
    
    | PET Range | Risk Level | Work Restriction |
    |-----------|------------|------------------|
    | >41°C | Critical | Emergency — All work suspended |
    | 35-41°C | High | 45-min work / 15-min rest |
    | 29-35°C | Moderate | 60-min work / 10-min rest |
    | 23-29°C | Low | Standard operations |
    | <23°C | Comfortable | Normal operations |
    
    **AI Explanation System**
    
    The AI assistant analyzes your specific site conditions and provides:
    - **WHY** this risk level exists (combination of factors)
    - **WHAT** factors matter most (ranked by impact)
    - **HOW** to address each factor (specific recommendations)
    - **PRACTICAL** guidance for site management
    
    Supported LLM providers:
    - OpenAI (GPT-3.5, GPT-4)
    - DeepSeek
    
    **Productivity Loss Calculation**
    
    Based on empirical models from construction site studies:

    Where:
- PL = Productivity loss (%)
- ΔPET = PET above work-specific alert level (°C)
- k = Work type coefficient (0.15-0.30)
""")

st.markdown("""
<div class="footer-text">
🧠 Construction Heat Stress AI Assistant | Powered by LLM | Version 1.0
</div>
""", unsafe_allow_html=True)
