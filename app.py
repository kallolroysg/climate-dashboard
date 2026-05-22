import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib
import os
import json

# --- PAGE SETUP & SESSION STATE ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")
st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("Powered by Our World in Data, XGBoost, and Google Gemini.")

# We create "Session States" to let the AI control the sliders
if 'ren_val' not in st.session_state: st.session_state.ren_val = 0
if 'ev_val' not in st.session_state: st.session_state.ev_val = 0
if 'tax_val' not in st.session_state: st.session_state.tax_val = 0

# --- CONFIGURE AI ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-3.5-flash') 
else:
    st.warning("⚠️ AI API Key not found in Streamlit Secrets. The AI Chatbot will be disabled.")

# --- DATA LOADING ---
@st.cache_resource
def load_ml_assets():
    model_file = 'best_co2_model.joblib'
    data_file = 'cleaned_sg_co2_data.csv'
    
    if not os.path.exists(model_file) or not os.path.exists(data_file):
        st.error(f"⚠️ Missing ML files! Please ensure '{model_file}' and '{data_file}' are uploaded to GitHub.")
        return None, None
    try:
        model = joblib.load(model_file)
        full_data = pd.read_csv(data_file)
        return model, full_data
    except Exception as e:
        st.error(f"⚠️ Error reading ML files: {e}")
        return None, None

ml_model, full_data = load_ml_assets()

# --- MAIN DASHBOARD ---
if ml_model is not None and full_data is not None:
    
    # --- SIDEBAR: POLICY SCENARIOS ---
    st.sidebar.header("🎛️ Policy Scenarios (Model Inputs)")
    
    # Notice the "key" argument. This ties the slider to the AI's brain!
    renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Fossil Fuel %)", 0, 50, key='ren_val', step=5)
    ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO₂ %)", 0, 50, key='ev_val', step=5)
    carbon_tax_intensity = st.sidebar.slider("Carbon Tax (Improve Energy/GDP %)", 0, 20, key='tax_val', step=2)

    st.sidebar.divider()
    st.sidebar.subheader("📊 Dataset & Model Overview")
    st.sidebar.caption("✅ Source: Our World in Data")
    st.sidebar.caption("✅ Target: CO2 Emissions")
    st.sidebar.caption("✅ Model: XGBoost Regressor")

    # --- ML PREDICTION ENGINE ---
    features_list = ['year', 'population', 'gdp', 'primary_energy_consumption', 'energy_per_gdp', 'energy_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'fossil_fuel_co2']
    
    hist_data = full_data.tail(10)
    hist_years = hist_data['year'].tolist()
    hist_co2 = hist_data['co2'].tolist()
    
    latest_features = full_data.tail(1).iloc[0]
    latest_year = int(latest_features['year'])
    
    future_years = list(range(latest_year + 1, 2036))
    baseline_preds = []
    policy_preds = []

    for year in future_years:
        base_row = latest_features.copy()
        base_row['year'] = year
        base_df = pd.DataFrame([base_row])[features_list]
        b_pred = ml_model.predict(base_df)[0]
        baseline_preds.append(b_pred)

        pol_row = base_row.copy()
        pol_row['oil_co2'] = pol_row['oil_co2'] * (1 - (ev_intensity/100))
        pol_row['fossil_fuel_co2'] = pol_row['fossil_fuel_co2'] * (1 - (renewable_intensity/100))
        pol_row['energy_per_gdp'] = pol_row['energy_per_gdp'] * (1 - (carbon_tax_intensity/100))
        
        pol_df = pd.DataFrame([pol_row])[features_list]
        p_pred = ml_model.predict(pol_df)[0]
        policy_preds.append(p_pred)

    # --- DASHBOARD METRICS ---
    idx_2030 = future_years.index(2030) if 2030 in future_years else -1
    pred_2030_base = baseline_preds[idx_2030]
    pred_2030_policy = policy_preds[idx_2030]
    reduction = pred_2030_base - pred_2030_policy

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline 2030 Forecast", f"{pred_2030_base:.2f} Mt")
    col2.metric("Policy-Adjusted 2030 Forecast", f"{pred_2030_policy:.2f} Mt", f"-{reduction:.2f} Mt", delta_color="inverse")
    if pred_2030_base > 0:
        col3.metric("Total Reduction (%)", f"{(reduction/pred_2030_base)*100:.1f}%")

    st.write("---")

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(["📈 ML Forecast vs Scenarios", "🤖 Smart AI Policy Engine"])

    with tab1:
        st.subheader("Predictive ML Forecast (Historical + Future to 2035)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_years, y=hist_co2, mode='lines+markers', name='Historical Data', line=dict(color='black', width=2)))
        
        connect_year = [hist_years[-1], future_years[0]]
        connect_base_co2 = [hist_co2[-1], baseline_preds[0]]
        connect_pol_co2 = [hist_co2[-1], policy_preds[0]]

        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_base_co2 + baseline_preds[1:], mode='lines', name='Baseline Forecast', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_pol_co2 + policy_preds[1:], mode='lines', name='Policy Intervention', line=dict(color='green', width=3)))
        
        fig.update_layout(xaxis_title="Year", yaxis_title="CO₂ Emissions (Mt)", height=450, hovermode="x unified", xaxis=dict(type='category'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Automated Dashboard Controller")
        st.caption("Type a real-world policy idea. The AI will classify it and automatically adjust the sliders on the left to simulate the impact.")
        
        user_policy = st.text_area("Example: 'Let's ban all petrol cars by 2030 and invest heavily in solar panels.'")
        
        if st.button("Simulate Policy Impact"):
            if not API_KEY:
                st.error("API Key missing.")
            else:
                # We ask the AI for the JSON code in the background
                prompt = f"""
                You are an internal data parser. 
                Classify the user's policy into one or more of these scenarios: "Renewable energy", "EV transition", or "Carbon tax".
                Determine the intensity: "low", "medium", or "high".
                Output ONLY valid JSON format exactly like this example, nothing else:
                {{
                  "scenario": "EV transition",
                  "intensity": "high"
                }}
                
                User Policy: {user_policy}
                """
                
                try:
                    response = model_ai.generate_content(prompt)
                    # Clean up the AI's response to get pure JSON code
                    clean_json = response.text.replace("```json","").replace("```","").strip()
                    parsed_data = json.loads(clean_json)
                    
                    scenario = parsed_data.get("scenario", "").lower()
                    intensity = parsed_data.get("intensity", "low").lower()
                    
                    # Convert the AI's "intensity" word into numbers for our sliders
                    val_50_scale = 10 if intensity == "low" else 25 if intensity == "medium" else 50
                    val_20_scale = 5 if intensity == "low" else 10 if intensity == "medium" else 20
                    
                    # Reset sliders to 0 first
                    st.session_state.ren_val = 0
                    st.session_state.ev_val = 0
                    st.session_state.tax_val = 0
                    
                    # Automatically move the sliders based on what the AI decided
                    if "ev" in scenario or "petrol" in scenario:
                        st.session_state.ev_val = val_50_scale
                    if "renewable" in scenario or "solar" in scenario:
                        st.session_state.ren_val = val_50_scale
                    if "tax" in scenario or "carbon" in scenario:
                        st.session_state.tax_val = val_20_scale
                    
                    # Tell Streamlit to refresh the page to show the new chart!
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to process policy. Please try phrasing it differently.")
