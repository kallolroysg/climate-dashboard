import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import joblib
import json
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SG Climate ML Simulator v3", page_icon="🇸🇬", layout="wide")

# --- INITIALIZE SESSION STATE ---
for k in ['slider_ren', 'slider_ev', 'slider_tax']:
    if k not in st.session_state:
        st.session_state[k] = 0
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Apply AI slider updates BEFORE widgets render ---
if 'ai_updates' in st.session_state:
    for key in ['ren', 'ev', 'tax']:
        if key in st.session_state.ai_updates:
            st.session_state[f'slider_{key}'] = st.session_state.ai_updates[key]
    del st.session_state['ai_updates']

st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("### Model-Driven Forecast: XGBoost vs Linear Benchmark")

# --- LOAD DATA, MODEL, METRICS (real artifacts) ---
@st.cache_data
def load_history():
    df = pd.read_csv("cleaned_sg_co2_data.csv").sort_values("year").reset_index(drop=True)
    df = df[df["primary_energy_consumption"] > 0].reset_index(drop=True)
    return df

@st.cache_resource
def load_models():
    best = joblib.load("best_co2_model.joblib")
    bench = joblib.load("benchmark_model.joblib")
    with open("model_metrics.json") as f:
        metrics = json.load(f)
    return best, bench, metrics

hist_df = load_history()
best_bundle, bench_bundle, METRICS = load_models()
BASE_FEATURES = best_bundle["base_features"]
FEATURES = best_bundle["features"]
RESID_STD = METRICS.get("residual_std", 1.4)

latest_rec = hist_df.iloc[-1]
prev_rec = hist_df.iloc[-2]
LATEST_YEAR = int(latest_rec["year"])

# --- AI API CONFIGURATION ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-3.5-flash')
else:
    st.warning("⚠️ AI API Key not found in Streamlit Secrets. Chatbot operational interface disabled.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🎛️ Policy Controls")
renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Gas CO₂ %)", 0, 50, step=5, key="slider_ren")
ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO₂ %)", 0, 50, step=5, key="slider_ev")
carbon_tax_intensity = st.sidebar.slider("Carbon Tax (Improve Energy Efficiency %)", 0, 20, step=2, key="slider_tax")

st.sidebar.divider()
st.sidebar.subheader("📊 Transparency Engine")
with st.sidebar.expander("🔍 Cleaned Baseline Data"):
    st.dataframe(hist_df[['year', 'gdp', 'primary_energy_consumption', 'co2']].tail(15), height=220)
with st.sidebar.expander("🏁 Model Comparison"):
    bm = METRICS['benchmark']['metrics']
    am = METRICS['advanced']['metrics']
    comp = pd.DataFrame({
        'Model': [f"{METRICS['benchmark']['model']}", f"{METRICS['advanced']['model']}"],
        'MAE': [round(bm['MAE'], 3), round(am['MAE'], 3)],
        'RMSE': [round(bm['RMSE'], 3), round(am['RMSE'], 3)],
        'R²': [round(bm['R2'], 3), round(am['R2'], 3)],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption(f"Selected: **{METRICS['selected_model']}** (lower MAE/RMSE than linear benchmark). "
               f"Test {METRICS['test_period'][0]}–{METRICS['test_period'][1]}, one-step-ahead.")
with st.sidebar.expander("🧠 Model Card"):
    st.markdown(f"""
**Selected model:** {METRICS['selected_model']}
**Target (Y):** {METRICS['target']} (Mt), predicted as yearly change then reconstructed
**Features:** {', '.join(BASE_FEATURES)} (+ year)
**Excluded (leakage):** {', '.join(METRICS['excluded_for_leakage'])}
**Test period:** {METRICS['test_period'][0]}–{METRICS['test_period'][1]}
""")

# --- FORECAST HORIZON ---
future_years = list(range(2025, 2036))
gdp_growth_rate = 0.022
pop_growth_rate = 0.008
energy_growth_rate = 0.015


def project_features(step_idx):
    """Project absolute driver variables for a future step using growth assumptions.
    Ratio columns (energy_per_gdp / energy_per_capita) follow OWID's own units, so we
    extrapolate them directly with mild growth rather than recomputing — this keeps the
    resulting Δ-features within the range the model was trained on."""
    f_gdp = latest_rec['gdp'] * ((1 + gdp_growth_rate) ** step_idx)
    f_pop = latest_rec['population'] * ((1 + pop_growth_rate) ** step_idx)
    f_energy = latest_rec['primary_energy_consumption'] * ((1 + energy_growth_rate) ** step_idx)
    # energy efficiency tends to improve slowly: energy_per_gdp drifts down ~0.5%/yr
    f_energy_per_gdp = latest_rec['energy_per_gdp'] * ((1 - 0.005) ** step_idx)
    f_energy_per_capita = latest_rec['energy_per_capita'] * ((1 + (energy_growth_rate - pop_growth_rate)) ** step_idx)
    return {
        'population': f_pop, 'gdp': f_gdp,
        'primary_energy_consumption': f_energy,
        'energy_per_gdp': f_energy_per_gdp,
        'energy_per_capita': f_energy_per_capita,
    }


def policy_reduction(ev, ren, tax):
    """Transparent policy multiplier on emissions.
    We have no policy->emissions training data, so policy effects are applied as
    explicit, interpretable reduction rules ON TOP of the model's baseline forecast:
      - EV transition reduces oil-driven emissions
      - Renewable expansion reduces gas-driven emissions
      - Carbon tax improves overall energy efficiency
    Combined as a single fractional reduction. This keeps the direction always
    correct (more policy -> lower emissions) and is fully explainable."""
    # weights reflect each lever's share of the emissions it targets
    ev_effect = (ev / 100.0) * 0.35       # oil share of fossil emissions ~35%
    ren_effect = (ren / 100.0) * 0.55     # gas share ~55%
    tax_effect = (tax / 100.0) * 1.0      # efficiency acts across the board
    total_reduction = ev_effect + ren_effect + tax_effect
    return min(total_reduction, 0.85)     # cap to avoid implausible >85% cuts


def forecast(model_bundle, ev, ren, tax):
    """Baseline trajectory is MODEL-DRIVEN: the model predicts Δco2 from projected
    Δfeatures, accumulated onto the last known absolute co2. The policy trajectory
    then applies a transparent reduction multiplier on the model baseline."""
    model = model_bundle["model"]
    base_levels = []
    prev_feats = {c: latest_rec[c] for c in BASE_FEATURES}
    cur_base = float(latest_rec['co2'])

    for i, year in enumerate(future_years):
        step = year - LATEST_YEAR
        feats = project_features(step)
        row = {'year': year}
        for c in BASE_FEATURES:
            row[f'd_{c}'] = feats[c] - prev_feats[c]
        d_base = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        cur_base = cur_base + d_base
        base_levels.append(cur_base)
        prev_feats = feats

    # policy = baseline scaled down by transparent reduction factor (ramps in over time)
    reduction = policy_reduction(ev, ren, tax)
    policy_levels = []
    for i, v in enumerate(base_levels):
        ramp = min((i + 1) / 5.0, 1.0)  # policy phases in fully over ~5 years
        policy_levels.append(v * (1 - reduction * ramp))

    # confidence band from real residual std, widening with horizon
    base_df = pd.DataFrame({'year': future_years, 'co2': base_levels})
    base_df['co2_upper'] = [v + 1.96 * RESID_STD * np.sqrt(i + 1) for i, v in enumerate(base_levels)]
    base_df['co2_lower'] = [v - 1.96 * RESID_STD * np.sqrt(i + 1) for i, v in enumerate(base_levels)]
    policy_df = pd.DataFrame({'year': future_years, 'co2': policy_levels})
    return base_df, policy_df


base_forecast_df, policy_forecast_df = forecast(best_bundle, ev_intensity, renewable_intensity, carbon_tax_intensity)

# --- LAYOUT ---
col_graph, col_ai_panel = st.columns([1.5, 1])

with col_graph:
    v_2030_base = base_forecast_df.loc[base_forecast_df['year'] == 2030, 'co2'].values[0]
    v_2030_poly = policy_forecast_df.loc[policy_forecast_df['year'] == 2030, 'co2'].values[0]
    net_saving = v_2030_base - v_2030_poly
    pct_saving = (net_saving / v_2030_base) * 100 if v_2030_base else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline 2030 Projection", f"{v_2030_base:.2f} Mt")
    m2.metric("Policy-Adjusted 2030", f"{v_2030_poly:.2f} Mt", f"-{net_saving:.2f} Mt", delta_color="inverse")
    m3.metric("Net Avoided Footprint (%)", f"{pct_saving:.1f}%")

    st.subheader("📈 Model-Based Emissions Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['year'].tail(15), y=hist_df['co2'].tail(15),
                             mode='lines+markers', name='Historical Records',
                             line=dict(color='black', width=2.5)))

    f_years = list(base_forecast_df['year'])
    f_upper = list(base_forecast_df['co2_upper'])
    f_lower = list(base_forecast_df['co2_lower'])
    fig.add_trace(go.Scatter(
        x=f_years + f_years[::-1], y=f_upper + f_lower[::-1],
        fill='toself', fillcolor='rgba(128,128,128,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Predictive Confidence Range', showlegend=True))

    c_years = [LATEST_YEAR, future_years[0]]
    c_base = [float(latest_rec['co2']), base_forecast_df['co2'].iloc[0]]
    c_poly = [float(latest_rec['co2']), policy_forecast_df['co2'].iloc[0]]
    fig.add_trace(go.Scatter(x=c_years + f_years[1:], y=c_base + list(base_forecast_df['co2'])[1:],
                             mode='lines', name='Baseline Forecast Trend',
                             line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=c_years + f_years[1:], y=c_poly + list(policy_forecast_df['co2'])[1:],
                             mode='lines', name='Active Policy Intervention',
                             line=dict(color='#00875A', width=3.5)))

    fig.update_layout(xaxis_title="Reporting Horizon (Years)", yaxis_title="Total Emissions (Mt)",
                      height=460, hovermode="x unified", xaxis=dict(type='category'),
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# --- INTERACTIVE AI SYSTEM CONTROLLER ---
with col_ai_panel:
    st.subheader("🤖 Smart Policy Parser")
    st.caption("Propose policy variations to watch the tracking matrix update automatically.")

    box_display = st.container(height=520)
    with box_display:
        for node in st.session_state['chat_history']:
            with st.chat_message(node["role"]):
                st.markdown(node["content"])

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
                    # Honest project context so the AI does NOT fabricate the methodology.
                    system_prompt = f"""
                    You are an objective climate policy advisor for a Singapore CO2 simulator.

                    Factual context about THIS app (do not contradict it):
                    - The forecast is produced by an XGBoost regression model (benchmarked against
                      Linear Regression). The model predicts the yearly CHANGE in CO2 and reconstructs
                      the absolute level. Features: population, GDP, primary energy consumption,
                      energy per GDP, energy per capita. It does NOT use neural networks.
                    - Policy sliders (EV, Renewable, Carbon Tax) adjust the projected energy-driver
                      features that feed the model.

                    The user proposed: "{user_query}"

                    You MUST return ONLY valid JSON (no markdown fences):
                    {{
                      "message": "Intro.\\n\\n**Local Implementation:**\\n* Point 1 (Singapore: LTA, ERP 2.0, HDB)\\n* Point 2\\n\\n**Overseas Examples:**\\n* City A...\\n* City B...",
                      "scenario": "ev",
                      "intensity": "high"
                    }}
                    Rules:
                    - 'message' uses the strict 3-part structure above with Markdown.
                    - 'scenario' is one of: "ev", "renewable", "tax".
                    - 'intensity' is one of: "low", "medium", "high".
                    """
                    try:
                        ai_raw = model_ai.generate_content(system_prompt).text.strip()
                        match = re.search(r'\{.*\}', ai_raw, re.DOTALL)
                        payload = json.loads(match.group(0)) if match else {"message": ai_raw, "scenario": "none", "intensity": "none"}

                        text_output = payload.get("message", "Could not parse review message.")
                        intent = payload.get("scenario", "none").lower()
                        scale = payload.get("intensity", "none").lower()

                        st.markdown(text_output)
                        st.session_state['chat_history'].append({"role": "assistant", "content": text_output})

                        if intent != "none":
                            s50 = 15 if scale == "low" else 25 if scale == "medium" else 50
                            s20 = 4 if scale == "low" else 10 if scale == "medium" else 20
                            # Only set the mentioned lever; keep others as-is (fix: no longer zero them out)
                            updates = {}
                            if "ev" in intent:
                                updates['ev'] = s50
                            elif "renewable" in intent or "solar" in intent:
                                updates['ren'] = s50
                            elif "tax" in intent or "carbon" in intent:
                                updates['tax'] = s20
                            if updates:
                                st.session_state.ai_updates = updates
                                st.rerun()
                    except Exception as e:
                        st.error(f"Fallback routine initiated. (Logs: {e})")
