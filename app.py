import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SG Climate ML Simulator v2", page_icon="🇸🇬", layout="wide")

# --- INITIALIZE SESSION STATE ---
# FIX: Direct Initialization of Streamlit Widget Keys
if 'slider_ren' not in st.session_state:
    st.session_state['slider_ren'] = 0
if 'slider_ev' not in st.session_state:
    st.session_state['slider_ev'] = 0
if 'slider_tax' not in st.session_state:
    st.session_state['slider_tax'] = 0
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("### Refactored Architecture: Mathematically Bound Downstream Engine")

# --- AI API CONFIGURATION ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    # Using the standard flash model name
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.warning("⚠️ AI API Key not found in Streamlit Secrets. Chatbot operational interface disabled.")

# --- DATA GENERATION & SEED MATRIX ---
@st.cache_data
def get_historical_repository():
    years = list(range(2015, 2025))
    data = {
        'year': years,
        'population': [5535000, 5607000, 5612000, 5638000, 5703000, 5685000, 5454000, 5637000, 5917000, 6040000],
        'gdp': [2.8e11, 2.9e11, 3.1e11, 3.2e11, 3.3e11, 3.2e11, 3.4e11, 3.6e11, 3.8e11, 3.9e11],
        'primary_energy_consumption': [95.0, 97.2, 101.4, 103.1, 105.6, 102.1, 99.8, 104.2, 108.5, 110.2],
        'oil_co2': [18.2, 18.5, 19.1, 19.4, 19.8, 18.2, 17.1, 18.0, 18.9, 19.2],
        'gas_co2': [25.1, 25.8, 26.4, 27.0, 27.5, 28.1, 28.4, 29.0, 29.6, 30.1],
        'coal_co2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    df['energy_per_gdp'] = df['primary_energy_consumption'] / (df['gdp'] / 1e11)
    df['fossil_fuel_co2'] = df['coal_co2'] + df['oil_co2'] + df['gas_co2']
    df['co2'] = df['fossil_fuel_co2'] + 1.2
    return df

hist_df = get_historical_repository()
latest_rec = hist_df.iloc[-1]

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🎛️ Policy Controls")

# FIX: Removed the conflicting 'value=' argument. Streamlit now reads directly from the Session State 'key'
renewable_intensity = st.sidebar.slider(
    "Renewable Expansion (Reduce Gas CO₂ %)", 
    0, 50, step=5, key="slider_ren"
)
ev_intensity = st.sidebar.slider(
    "EV Transition (Reduce Oil CO₂ %)", 
    0, 50, step=5, key="slider_ev"
)
carbon_tax_intensity = st.sidebar.slider(
    "Carbon Tax (Improve Energy Efficiency %)", 
    0, 20, step=2, key="slider_tax"
)

st.sidebar.divider()
st.sidebar.subheader("📊 Transparency Engine")
with st.sidebar.expander("🔍 Cleaned Baseline Data"):
    st.dataframe(hist_df[['year', 'gdp', 'primary_energy_consumption', 'co2']], height=220)

# --- HYBRID PREDICTION SYSTEM ---
future_years = list(range(2025, 2036))
gdp_growth_rate = 0.022
pop_growth_rate = 0.008
energy_growth_rate = 0.015
σ_variance = 0.025  

def run_forecast_simulation(ev, ren, tax):
    baseline_records = []
    policy_records = []
    
    for i, year in enumerate(future_years):
        step_idx = year - int(latest_rec['year'])
        uncertainty_factor = σ_variance * np.sqrt(step_idx)
        
        b_gdp = latest_rec['gdp'] * ((1 + gdp_growth_rate) ** step_idx)
        b_pop = latest_rec['population'] * ((1 + pop_growth_rate) ** step_idx)
        b_energy = latest_rec['primary_energy_consumption'] * ((1 + energy_growth_rate) ** step_idx)
        
        b_oil = latest_rec['oil_co2'] * ((1 + 0.005) ** step_idx)
        b_gas = latest_rec['gas_co2'] * ((1 + 0.01) ** step_idx)
        b_coal = 0.0
        
        b_energy_per_gdp = b_energy / (b_gdp / 1e11)
        b_fossil = b_coal + b_oil + b_gas
        b_total_co2 = b_fossil + 1.2
        
        baseline_records.append({
            'year': year, 'gdp': b_gdp, 'population': b_pop,
            'co2': b_total_co2, 'co2_upper': b_total_co2 * (1 + 1.96 * uncertainty_factor),
            'co2_lower': b_total_co2 * (1 - 1.96 * uncertainty_factor)
        })
        
        p_gdp = b_gdp
        p_energy_per_gdp = b_energy_per_gdp * (1 - (tax / 100))
        p_energy = p_energy_per_gdp * (p_gdp / 1e11)
        efficiency_scale = p_energy / b_energy
        
        p_oil = (b_oil * efficiency_scale) * (1 - (ev / 100))
        displaced_oil_co2 = (b_oil * efficiency_scale) - p_oil
        induced_gas_load = displaced_oil_co2 * 0.35
        
        p_gas_base = (b_gas * efficiency_scale) + induced_gas_load
        p_gas = p_gas_base * (1 - (ren / 100))
        p_coal = 0.0
        
        p_fossil = p_coal + p_oil + p_gas
        p_total_co2 = p_fossil + 1.2
        
        policy_records.append({
            'year': year, 'co2': p_total_co2,
            'co2_upper': p_total_co2 * (1 + 1.96 * uncertainty_factor),
            'co2_lower': p_total_co2 * (1 - 1.96 * uncertainty_factor)
        })
        
    return pd.DataFrame(baseline_records), pd.DataFrame(policy_records)

base_forecast_df, policy_forecast_df = run_forecast_simulation(ev_intensity, renewable_intensity, carbon_tax_intensity)

# --- LAYOUT SEGMENTATION ---
col_graph, col_ai_panel = st.columns([1.5, 1])

with col_graph:
    v_2030_base = base_forecast_df.loc[base_forecast_df['year'] == 2030, 'co2'].values[0]
    v_2030_poly = policy_forecast_df.loc[policy_forecast_df['year'] == 2030, 'co2'].values[0]
    net_saving = v_2030_base - v_2030_poly
    pct_saving = (net_saving / v_2030_base) * 100
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Baseline 2030 Projection", f"{v_2030_base:.2f} Mt")
    m_col2.metric("Policy-Adjusted 2030", f"{v_2030_poly:.2f} Mt", f"-{net_saving:.2f} Mt", delta_color="inverse")
    m_col3.metric("Net Avoided Footprint (%)", f"{pct_saving:.1f}%")
    
    st.subheader("📈 Statistical Hybrid Emissions Trajectory")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=hist_df['year'], y=hist_df['co2'], mode='lines+markers', name='Historical Records', line=dict(color='black', width=2.5)))
    
    c_years = [hist_df['year'].iloc[-1], future_years[0]]
    c_base = [hist_df['co2'].iloc[-1], base_forecast_df['co2'].iloc[0]]
    c_poly = [hist_df['co2'].iloc[-1], policy_forecast_df['co2'].iloc[0]]
    
    f_years = list(base_forecast_df['year'])
    f_upper = list(base_forecast_df['co2_upper'])
    f_lower = list(base_forecast_df['co2_lower'])
    
    # FIX: Ensure width=0 is used instead of 'transparent' to satisfy Plotly Validation
    fig.add_trace(go.Scatter(
        x=f_years + f_years[::-1],
        y=f_upper + f_lower[::-1],
        fill='toself',
        fillcolor='rgba(128,128,128,0.15)',
        line=dict(width=0), 
        name='95% Predictive Confidence Range',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(x=c_years + f_years[1:], y=c_base + list(base_forecast_df['co2'])[1:], mode='lines', name='Baseline Forecast Trend', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=c_years + f_years[1:], y=c_poly + list(policy_forecast_df['co2'])[1:], mode='lines', name='Active Policy Intervention', line=dict(color='#00875A', width=3.5)))
    
    fig.update_layout(
        xaxis_title="Reporting Horizon (Years)",
        yaxis_title="Total Emissions (Mt)",
        height=480,
        hovermode="x unified",
        xaxis=dict(type='category'),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- INTERACTIVE AI SYSTEM CONTROLLER ---
with col_ai_panel:
    st.subheader("🤖 Smart Policy Parser")
    st.caption("Propose policy variations to watch the tracking matrix update automatically.")
    
    box_display = st.container(height=520)
    with box_display:
        for text_node in st.session_state['chat_history']:
            with st.chat_message(text_node["role"]):
                st.markdown(text_node["content"])
                
    if user_query := st.chat_input("E.g., What if we accelerate solar deployment and enforce carbon pricing?"):
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        with box_display:
            with st.chat_message("user"):
                st.markdown(user_query)
                
        with box_display:
            with st.chat_message("assistant"):
                if not API_KEY:
                    st.error("API configuration block is unavailable.")
                else:
                    # FIX: Enforce Strict JSON output so Streamlit can read the intent
                    system_prompt = f"""
                    You are an elite, objective climate policy advisor.
                    The user proposed: "{user_query}"
                    
                    You MUST return ONLY a valid JSON structure. Do NOT wrap it in markdown block quotes. Just the raw JSON.
                    {{
                      "message": "Your professional review commentary here. Discuss feasibility and practical steps...",
                      "scenario": "ev", 
                      "intensity": "high"
                    }}
                    
                    Rules for keys:
                    - 'scenario' MUST be one of: "ev", "renewable", "tax".
                    - 'intensity' MUST be one of: "low", "medium", "high".
                    """
                    try:
                        ai_raw = model_ai.generate_content(system_prompt).text.strip()
                        clean_match = re.search(r'\{.*\}', ai_raw, re.DOTALL)
                        
                        if clean_match:
                            parsed_payload = json.loads(clean_match.group(0))
                        else:
                            parsed_payload = {"message": ai_raw, "scenario": "none", "intensity": "none"}
                            
                        text_output = parsed_payload.get("message", "Could not parse review message.")
                        intent_label = parsed_payload.get("scenario", "none").lower()
                        scale_label = parsed_payload.get("intensity", "none").lower()
                        
                        st.markdown(text_output)
                        st.session_state['chat_history'].append({"role": "assistant", "content": text_output})
                        
                        if intent_label != "none":
                            fixed_50_scale = 15 if scale_label == "low" else 25 if scale_label == "medium" else 50
                            fixed_20_scale = 4 if scale_label == "low" else 10 if scale_label == "medium" else 20
                            
                            # FIX: Modify Streamlit widget keys directly to force the UI to refresh
                            if "ev" in intent_label:
                                st.session_state['slider_ren'] = 0
                                st.session_state['slider_tax'] = 0
                                st.session_state['slider_ev'] = fixed_50_scale
                            elif "renewable" in intent_label or "solar" in intent_label:
                                st.session_state['slider_ev'] = 0
                                st.session_state['slider_tax'] = 0
                                st.session_state['slider_ren'] = fixed_50_scale
                            elif "tax" in intent_label or "carbon" in intent_label:
                                st.session_state['slider_ev'] = 0
                                st.session_state['slider_ren'] = 0
                                st.session_state['slider_tax'] = fixed_20_scale
                                
                            st.rerun()
                            
                    except Exception as fatal_error:
                        st.error(f"Fallback routine initiated. Interface reset required. (Logs: {fatal_error})")
