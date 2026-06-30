"""
TR Heat Stress Advisor — LLM-powered Construction Heat Stress Assistant
=========================================================================
A Streamlit app where users describe a work scenario in natural language
(or fill in parameters), an LLM (Claude) interprets it against NIOSH/OSHA/
ACGIH heat-stress guidance, and the app renders structured recommendations
plus Plotly visualizations (WBGT gauge, REL work/rest chart, risk timeline).

Deploy: GitHub + Streamlit Community Cloud.
Free LLM backends supported: Groq (Llama 3.3 70B) and Google Gemini (free tier).
Secrets required (at least one): GROQ_API_KEY and/or GEMINI_API_KEY
(Settings -> Secrets on Streamlit Cloud, a local .streamlit/secrets.toml file,
or environment variables).
"""

import os
import json
import re
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ----------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TR Heat Stress Advisor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg,#0f2027 0%,#203a43 50%,#2c5364 100%);}
    .main-header {font-size:2.4rem;font-weight:700;text-align:center;color:white;
        padding:1.2rem;background:rgba(255,255,255,0.08);backdrop-filter:blur(12px);
        border-radius:20px;margin-bottom:1.2rem;border:1px solid rgba(255,255,255,0.1);}
    .risk-VERY-HIGH {background:linear-gradient(135deg,#8B0000,#FF0000);}
    .risk-HIGH {background:linear-gradient(135deg,#FF4500,#FF8C00);}
    .risk-MODERATE {background:linear-gradient(135deg,#FFA500,#FFD700);color:#222;}
    .risk-LOW {background:linear-gradient(135deg,#006400,#228B22);}
    .risk-box {padding:1.2rem;border-radius:16px;color:white;text-align:center;margin:.6rem 0;}
    .risk-box h2 {margin:0;font-size:1.8rem;}
    .rec-card {background:rgba(255,255,255,0.07);backdrop-filter:blur(8px);padding:1.1rem;
        border-radius:14px;color:white;border:1px solid rgba(255,255,255,0.1);height:100%;}
    .rec-card h4 {margin-top:0;border-bottom:2px solid rgba(255,255,255,0.2);padding-bottom:.4rem;}
    .source-pill {display:inline-block;background:rgba(255,255,255,0.12);color:#9be8e0;
        padding:2px 10px;border-radius:12px;font-size:11px;margin:2px 4px 2px 0;}
    [data-testid="stChatMessage"] {background:rgba(255,255,255,0.05);border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🌡️ TR Heat Stress Advisor — AI Assistant</div>', unsafe_allow_html=True)
st.caption(
    "Describe a work scenario in plain language (e.g. *'It's 35°C and 70% humidity, "
    "crew is doing heavy rebar tying in direct sun, what should we do?'*) or fill in the "
    "sidebar parameters. The assistant reasons using NIOSH, OSHA, and ACGIH heat-stress "
    "guidance and returns a risk assessment, recommendations, and visualizations."
)

# ----------------------------------------------------------------------
# KNOWLEDGE BASE — condensed NIOSH / OSHA / ACGIH guidance
# This is embedded in the system prompt so the LLM grounds its answers
# in real standards instead of hallucinating thresholds.
# ----------------------------------------------------------------------
HEAT_STRESS_KNOWLEDGE = r"""
You are TR Heat Stress Advisor, an expert occupational safety assistant specializing in
heat stress in construction and outdoor manual labor. Ground every answer in the
following condensed reference material (NIOSH, OSHA, ACGIH). Cite the source body
(NIOSH / OSHA / ACGIH) inline when you state a threshold or recommendation.

=== 1. WBGT (Wet Bulb Globe Temperature) — NIOSH / ACGIH Work-Rest & TLV framework ===
WBGT (°C) is the primary metric for heat-stress action levels. Work is categorized by
metabolic workload into four classes, each with its own WBGT Recommended Alert Limit
(REL, unacclimatized) and Threshold Limit Value (TLV, acclimatized), assuming standard
light clothing (clo ~0.6) and ~25%/75% to 100% work/rest cycles as applicable:
  - Light work (~200 W / ~2.0-2.5 met, e.g. sitting/standing light hand work):
      Unacclimatized REL ≈ 27.5°C WBGT (100% work); Acclimatized TLV ≈ 29.5°C WBGT
  - Moderate work (~300 W / ~3.0 met, e.g. walking, moderate lifting):
      Unacclimatized REL ≈ 26.0°C WBGT (100% work); Acclimatized TLV ≈ 27.5°C WBGT
  - Heavy work (~400 W / ~3.5-4.0 met, e.g. shoveling, heavy lifting, rebar/formwork):
      Unacclimatized REL ≈ 25.0°C WBGT (100% work); Acclimatized TLV ≈ 26.0°C WBGT
  - Very heavy work (~500+ W / ~4.5+ met, e.g. continuous heavy manual labor):
      Unacclimatized REL ≈ 23.0°C WBGT (100% work); Acclimatized TLV ≈ 25.0°C WBGT
As WBGT rises above these levels, NIOSH/ACGIH prescribe progressively shorter work
cycles with rest in the shade: 75%, 50%, then 25% work-per-hour tiers, derived from the
ACGIH TLV work/rest table. Add ~+1°C WBGT credit for light/permeable clothing acclimatized
workers in good condition; subtract for impermeable PPE.

=== 2. OSHA Heat Illness Prevention Program (General Duty Clause + NEP) ===
OSHA does not (as of the assistant's training) have a single finalized federal heat
standard but enforces heat-illness prevention via the General Duty Clause and a National
Emphasis Program. Core required/expected elements:
  - Water: cool drinking water accessible within a short walk of every work area.
  - Rest: shaded or air-conditioned rest areas; scheduled rest breaks scaled to heat risk.
  - Shade: required when temperatures approach/exceed ~80°F (27°C) heat index per many
    state plans (e.g., Cal/OSHA, Oregon OSHA, Washington L&I) used as reference practice.
  - Acclimatization: new or returning workers build heat tolerance gradually — no more
    than ~20% of normal workload/duration on day 1, increasing ~20%/day over 7-14 days.
  - Training: supervisors and workers trained to recognize heat illness signs/symptoms.
  - High-Heat Procedures (state-plan model, e.g. Cal/OSHA at ≥95°F / ~35°C): mandatory
    10-min paid cool-down break every 2 hours, mandatory buddy system / observation,
    pre-shift safety meeting.
  - Emergency response: clear procedure to summon help for suspected heat stroke;
    heat stroke is a medical emergency (call 911 / EMS, active cooling, do not wait).

=== 3. NIOSH Heat Index / Risk-Level Tiers (parallel framework, °F/°C) ===
  - Caution (80-90°F / 27-32°C heat index): fatigue possible with prolonged exposure.
  - Extreme Caution (90-103°F / 32-39°C): heat cramps/exhaustion possible.
  - Danger (103-124°F / 39-51°C): heat cramps/exhaustion likely, heat stroke possible
    with prolonged exposure/activity.
  - Extreme Danger (>124°F / >51°C): heat stroke highly likely.

=== 4. Heat illness spectrum (for explaining "why") ===
  - Heat rash / heat cramps: earliest signs, muscle cramps from electrolyte loss.
  - Heat syncope: fainting, often early in unacclimatized workers.
  - Heat exhaustion: heavy sweating, weakness, cool/clammy skin, nausea, headache,
    dizziness — core temp usually <40°C. Move to cool area, hydrate, remove excess
    clothing; if not improving in 30 min, seek medical care.
  - Heat stroke: core temp >40°C (104°F), altered mental status, hot/dry OR sweaty skin,
    rapid pulse — MEDICAL EMERGENCY. Call EMS immediately, cool aggressively
    (ice water immersion is gold standard), do not give fluids if unconscious.

=== 5. Productivity-loss framing (engineering/management context) ===
Construction productivity loss tends to follow a roughly exponential-saturating curve
above each work type's "allowable" thermal threshold (PET or WBGT alert level): minimal
loss near the threshold, rising sharply and saturating around 25-30% loss in the most
severe sustained-heat conditions for heavy manual tasks. Lighter cognitive/seated tasks
degrade less from heat than heavy dynamic labor.

=== 6. Clothing & metabolic adjustments ===
  - clo insulation: 0.36 (light summer) to 1.0+ (coveralls/PPE); each ~0.1 clo above
    baseline behaves like a few-degree WBGT/PET penalty for heat dissipation.
  - met (metabolic rate): rest ~1.0-1.2 met; light work ~2.0-2.5; moderate ~2.6-3.5;
    heavy ~3.5-4.5; very heavy >4.5. Higher met sharply increases internal heat
    production and required evaporative cooling capacity.

=== YOUR TASK ===
Given a user's natural-language scenario or explicit parameters (temperature, relative
humidity, wind speed, sun/shade exposure, clothing, activity/metabolic level, duration,
acclimatization status, age/health factors if mentioned), you must:
  1. Extract/estimate the relevant parameters (state any assumptions explicitly).
  2. Estimate WBGT if not given (rough outdoor-sun approximation acceptable; state it's
     an estimate) and classify the work/rest tier and risk level.
  3. Classify overall risk as one of: LOW, MODERATE, HIGH, VERY HIGH.
  4. Explain WHY in terms of the specific thresholds crossed (NIOSH/OSHA/ACGIH), in
     plain, practical language a site safety manager or worker could act on immediately.
  5. Give concrete, role-specific recommendations (site manager actions; worker actions):
     hydration cadence, work/rest cycle, shade/cooling, PPE/clothing, monitoring/buddy
     system, and when to escalate/stop work.
  6. ALWAYS finish your reply with a single fenced ```json code block (and nothing after
     it) containing a structured object so the app can render charts, using this exact
     schema (use your best numeric estimates; use null only if truly unknowable):
{
  "risk_level": "LOW | MODERATE | HIGH | VERY HIGH",
  "wbgt_c": <number>,
  "heat_index_c": <number or null>,
  "temperature_c": <number>,
  "relative_humidity_pct": <number>,
  "wind_speed_ms": <number or null>,
  "clo": <number>,
  "met": <number>,
  "work_type": "Rest|Light|Moderate|Heavy|Very Heavy",
  "rel_threshold_wbgt_c": <number>,
  "tlv_threshold_wbgt_c": <number>,
  "recommended_work_pct_per_hour": <25|50|75|100>,
  "estimated_productivity_loss_pct": <number 0-30>,
  "manager_actions": ["...", "..."],
  "worker_actions": ["...", "..."],
  "warning_signs": ["...", "..."],
  "sources": ["NIOSH", "OSHA", "ACGIH"]
}
Do not include any other text after the JSON block. Keep the prose portion focused,
well-organized with short headers, and avoid being repetitive with the JSON.
"""

# ----------------------------------------------------------------------
# LLM CLIENTS — free-tier backends: Groq (Llama 3.3 70B) and Google Gemini
# ----------------------------------------------------------------------
def get_secret(name):
    val = os.environ.get(name)
    if not val:
        try:
            val = st.secrets.get(name, None)
        except Exception:
            val = None
    return val


def get_groq_key():
    return get_secret("GROQ_API_KEY")


def get_gemini_key():
    return get_secret("GEMINI_API_KEY")


def get_client(provider):
    """Return a ready-to-use client (or config) for the chosen provider, or None."""
    if provider == "Groq (Llama 3.3 70B — free)":
        key = get_groq_key()
        if not key or Groq is None:
            return None
        return Groq(api_key=key)
    elif provider == "Google Gemini (free tier)":
        key = get_gemini_key()
        if not key or genai is None:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=HEAT_STRESS_KNOWLEDGE,
        )
    return None


def ask_assistant(client, provider, history, max_tokens=1800):
    """Send full chat history + system prompt to the selected provider, return text."""
    user_assistant_msgs = [m for m in history if m["role"] in ("user", "assistant")]

    if provider == "Groq (Llama 3.3 70B — free)":
        messages = [{"role": "system", "content": HEAT_STRESS_KNOWLEDGE}]
        messages += [{"role": m["role"], "content": m["content"]} for m in user_assistant_msgs]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content

    elif provider == "Google Gemini (free tier)":
        # Gemini uses "user"/"model" roles and no separate system message in history
        gem_history = []
        for m in user_assistant_msgs[:-1]:
            role = "model" if m["role"] == "assistant" else "user"
            gem_history.append({"role": role, "parts": [m["content"]]})
        chat = client.start_chat(history=gem_history)
        last_msg = user_assistant_msgs[-1]["content"]
        response = chat.send_message(last_msg)
        return response.text

    return "⚠️ No provider configured."


def extract_json_block(text):
    """Pull the trailing ```json ... ``` block out of the model's reply."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{[^`]*\"risk_level\"[^`]*\})", text, re.DOTALL)
    if not match:
        return None, text
    raw = match.group(1)
    prose = text[: match.start()].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, text
    return data, prose


# ----------------------------------------------------------------------
# SIDEBAR — optional structured inputs (auto-fills a prompt for the chat)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Quick Scenario Builder")
    st.caption("Fill these in and click 'Ask about this scenario', or just type a question in the chat instead.")

    T = st.number_input("Temperature (°C)", 15.0, 50.0, 34.0, 0.5)
    RH = st.number_input("Relative Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 15.0, 1.5, 0.1)
    sun = st.selectbox("Exposure", ["Direct sun", "Partial shade", "Full shade / indoor"])
    clo = st.select_slider("Clothing insulation (clo)", options=[0.36, 0.5, 0.57, 0.61, 0.96, 1.0], value=0.57)
    met_label = st.selectbox(
        "Activity / work type",
        ["Rest (~1.2 met)", "Light (~2.2 met)", "Moderate (~3.0 met)", "Heavy (~3.8 met)", "Very Heavy (~4.5 met)"],
        index=3,
    )
    acclim = st.radio("Acclimatization", ["Acclimatized", "Unacclimatized / new worker"], horizontal=False)
    duration = st.slider("Continuous task duration (min)", 10, 480, 120, 10)

    if st.button("📋 Ask about this scenario", use_container_width=True):
        scenario_prompt = (
            f"Assess heat stress risk for this construction work scenario: "
            f"Temperature {T}°C, Relative Humidity {RH}%, Wind Speed {WS} m/s, "
            f"exposure: {sun}, clothing insulation {clo} clo, activity: {met_label}, "
            f"worker acclimatization: {acclim}, planned continuous task duration: "
            f"{duration} minutes. Give the risk level, the work/rest schedule, "
            f"and recommendations."
        )
        st.session_state["_pending_prompt"] = scenario_prompt

    st.markdown("---")
    st.markdown("## 🤖 LLM Backend")
    provider = st.selectbox(
        "Choose a free LLM backend",
        ["Groq (Llama 3.3 70B — free)", "Google Gemini (free tier)"],
        key="provider",
    )

    groq_present = bool(get_groq_key())
    gemini_present = bool(get_gemini_key())
    st.write(f"GROQ_API_KEY: {'✅ found' if groq_present else '❌ missing'}")
    st.write(f"GEMINI_API_KEY: {'✅ found' if gemini_present else '❌ missing'}")
    if (provider.startswith("Groq") and not groq_present) or (provider.startswith("Google") and not gemini_present):
        st.warning(
            "Selected backend's key is missing. Add it under **Settings → Secrets** "
            "on Streamlit Cloud, or as an environment variable locally:\n\n"
            "- Groq: get a free key at console.groq.com/keys → `GROQ_API_KEY = \"gsk_...\"`\n"
            "- Gemini: get a free key at aistudio.google.com/apikey → `GEMINI_API_KEY = \"...\"`"
        )
    else:
        st.success(f"{provider.split(' (')[0]} key detected ✅")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("last_data", None)
        st.rerun()


# ----------------------------------------------------------------------
# VISUALIZATION HELPERS
# ----------------------------------------------------------------------
RISK_COLORS = {"LOW": "#228B22", "MODERATE": "#FFD700", "HIGH": "#FF8C00", "VERY HIGH": "#FF0000"}


def render_risk_box(data):
    risk = data.get("risk_level", "MODERATE").upper().replace("_", " ")
    css_class = f"risk-{risk.replace(' ', '-')}"
    st.markdown(
        f"""<div class="risk-box {css_class}">
        <h2>🌡️ Risk Level: {risk}</h2>
        <p>WBGT estimate: <b>{data.get('wbgt_c', 'N/A')}°C</b> &nbsp;|&nbsp;
        Work type: <b>{data.get('work_type','N/A')}</b> &nbsp;|&nbsp;
        Recommended work/hour: <b>{data.get('recommended_work_pct_per_hour','N/A')}%</b></p>
        </div>""",
        unsafe_allow_html=True,
    )


def render_gauge(data):
    wbgt = data.get("wbgt_c") or 0
    tlv = data.get("tlv_threshold_wbgt_c") or 26
    rel = data.get("rel_threshold_wbgt_c") or 25
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=wbgt,
            number={"suffix": "°C", "font": {"color": "white"}},
            title={"text": "WBGT vs. Action Limits", "font": {"color": "white", "size": 16}},
            gauge={
                "axis": {"range": [15, 45], "tickcolor": "white"},
                "bar": {"color": "white", "thickness": 0.25},
                "steps": [
                    {"range": [15, rel], "color": "#228B22"},
                    {"range": [rel, tlv], "color": "#FFD700"},
                    {"range": [tlv, tlv + 3], "color": "#FF8C00"},
                    {"range": [tlv + 3, 45], "color": "#8B0000"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.9, "value": wbgt},
            },
        )
    )
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, margin=dict(t=60, b=10, l=20, r=20)
    )
    return fig


def render_workrest_bar(data):
    pct = data.get("recommended_work_pct_per_hour", 100)
    tiers = [100, 75, 50, 25]
    colors = ["#228B22" if t >= pct else "rgba(255,255,255,0.15)" for t in tiers]
    colors[tiers.index(pct)] = "#FF8C00" if pct < 100 else "#228B22"
    fig = go.Figure(
        go.Bar(
            x=[f"{t}% work/hr" for t in tiers],
            y=tiers,
            marker_color=colors,
            text=[f"{t}%" for t in tiers],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="ACGIH/NIOSH Work-Rest Cycle Tier",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        yaxis_title="% of each hour spent working",
        showlegend=False,
    )
    return fig


def render_productivity_chart(data):
    loss = data.get("estimated_productivity_loss_pct", 0) or 0
    work_types = ["Light", "Moderate", "Heavy", "Very Heavy"]
    base = {"Light": 0.4, "Moderate": 0.7, "Heavy": 1.0, "Very Heavy": 1.2}
    current = data.get("work_type", "Heavy").split()[0]
    vals = [round(loss * base.get(w, 1.0), 1) for w in work_types]
    colors = ["#FF0000" if w == current else "#4ECDC4" for w in work_types]
    fig = go.Figure(go.Bar(x=work_types, y=vals, marker_color=colors, text=[f"{v}%" for v in vals], textposition="auto"))
    fig.update_layout(
        title=f"Estimated Productivity Loss by Work Type (current: {current})",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        yaxis_title="Productivity loss (%)",
    )
    return fig


def render_recommendations(data):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="rec-card"><h4>🏗️ Site Manager Actions</h4>', unsafe_allow_html=True)
        for item in data.get("manager_actions", []) or ["No data."]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="rec-card"><h4>👷 Worker Actions</h4>', unsafe_allow_html=True)
        for item in data.get("worker_actions", []) or ["No data."]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("warning_signs"):
        st.markdown('<div class="rec-card" style="margin-top:.6rem;"><h4>🚨 Warning Signs to Watch For</h4>', unsafe_allow_html=True)
        for item in data["warning_signs"]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    sources = data.get("sources", [])
    if sources:
        pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
        st.markdown(f"**Sources referenced:** {pills}", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CHAT STATE
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_data" not in st.session_state:
    st.session_state["last_data"] = None

# Render past conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg.get("prose", msg["content"]))
            if msg.get("data"):
                render_risk_box(msg["data"])
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(render_gauge(msg["data"]), use_container_width=True, key=f"gauge_{msg['ts']}")
                with c2:
                    st.plotly_chart(render_workrest_bar(msg["data"]), use_container_width=True, key=f"wr_{msg['ts']}")
                st.plotly_chart(render_productivity_chart(msg["data"]), use_container_width=True, key=f"prod_{msg['ts']}")
                render_recommendations(msg["data"])

# Pull in sidebar-generated prompt if present
pending = st.session_state.pop("_pending_prompt", None)
user_input = st.chat_input("Describe a heat-stress scenario or ask a follow-up question...")
prompt_to_send = pending or user_input

if prompt_to_send:
    provider = st.session_state.get("provider", "Groq (Llama 3.3 70B — free)")
    client = get_client(provider)
    st.session_state["messages"].append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    with st.chat_message("assistant"):
        if client is None:
            error_msg = (
                f"I can't reach **{provider}** because no valid API key is configured "
                f"for it. Add the matching key under Streamlit Cloud **Settings → Secrets** "
                f"(or as a local environment variable), or switch providers in the sidebar."
            )
            st.error(error_msg)
            st.session_state["messages"].append({"role": "assistant", "content": error_msg, "prose": error_msg, "data": None, "ts": datetime.utcnow().isoformat()})
        else:
            with st.spinner("Analyzing against NIOSH / OSHA / ACGIH guidance..."):
                try:
                    raw_reply = ask_assistant(client, provider, st.session_state["messages"])
                except Exception as e:
                    raw_reply = f"⚠️ Error calling {provider}: {e}"
            data, prose = extract_json_block(raw_reply)
            st.markdown(prose if prose else raw_reply)
            ts = datetime.utcnow().isoformat()
            if data:
                render_risk_box(data)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(render_gauge(data), use_container_width=True, key=f"gauge_{ts}")
                with c2:
                    st.plotly_chart(render_workrest_bar(data), use_container_width=True, key=f"wr_{ts}")
                st.plotly_chart(render_productivity_chart(data), use_container_width=True, key=f"prod_{ts}")
                render_recommendations(data)
                st.session_state["last_data"] = data
            st.session_state["messages"].append(
                {"role": "assistant", "content": raw_reply, "prose": prose if prose else raw_reply, "data": data, "ts": ts}
            )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:rgba(255,255,255,0.6);font-size:13px;">
    TR Heat Stress Advisor combines Claude (Anthropic) reasoning with condensed
    NIOSH / OSHA / ACGIH heat-stress guidance. This tool supports — but does not
    replace — a qualified safety professional's judgment and your organization's
    heat illness prevention program. © 2026 Md. Tushar Ali, NJIT.
    </div>
    """,
    unsafe_allow_html=True,
)
