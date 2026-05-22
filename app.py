import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib
import os
import json
import re

# --- PAGE SETUP & SESSION STATE ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")

# 1. Safe Initialization 
if 'ren_val' not in st.session_state: st.session_state['ren_val'] = 0
if 'ev_val' not in st.session_state: st.session_state['ev_val'] = 0
if 'tax_val' not in st.session_state: st.session_state['tax_val'] = 0
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("Powered by Our World in Data, XGBoost, and Google Gemini.")

# --- CONFIGURE AI ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-2.5-flash') 
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
    
    st.sidebar.header("🎛️ Policy Scenarios (Model Inputs)")
    
    # 2. THE FIX: Removed rigid 'key' bindings. Now using dynamic 'value' bindings.
    # This prevents the "widget cannot be modified after instantiation" crash completely.
    renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Fossil Fuel %)", 0, 50, value=st.session_state['ren_val'], step=5)
    ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO₂ %)", 0, 50, value=st.session_state['ev_val'], step=5)
    carbon_tax_intensity = st.sidebar.slider("Carbon Tax (Improve Energy/GDP %)", 0, 20, value=st.session_state['tax_val'], step=2)

    # Manual updates sync to session state silently
    st.session_state['ren_val'] = renewable_intensity
    st.session_state['ev_val'] = ev_intensity
    st.session_state['tax_val'] = carbon_tax_intensity

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
    tab1, tab2 = st.tabs(["📈 ML Forecast vs Scenarios", "🤖 Smart AI Chat & Controller"])

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

        st.info("ℹ️ **Methodology Note:** These projections represent *scenario simulations* where input variables (e.g., oil CO₂) are adjusted prior to prediction. They do not constitute absolute causal proof, but illustrate expected trends based on historical ML patterns.")

    with tab2:
        st.subheader("Interactive AI Assistant")
        st.caption("Chat normally, or propose a policy! The AI acts ONLY as a classifier and will automatically adjust the dashboard sliders.")
        
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if user_input := st.chat_input("Ask a question, or say: 'Let's impose a high carbon tax.'"):
            
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            with st.chat_message("assistant"):
                if not API_KEY:
                    st.error("API Key missing.")
                else:
                    # 3. THE FIX: Ruthless JSON Prompt
                    prompt = f"""
                    You are a helpful climate policy AI. You do NOT make up mathematical predictions. 
                    The user said: "{user_input}"
                    
                    Task 1: If it's a general question (e.g., "Hello", "What is 2+2?"), answer it conversationally in the "message" field.
                    Task 2: If they propose a policy, explain it in the "message" field and state you are updating the dashboard parameters for the XGBoost model.
                    Task 3: Classify the policy into: "renewable", "ev", or "tax". If not a policy, output "none".
                    Task 4: Determine intensity: "low", "medium", or "high". If not a policy, output "none".
                    
                    CRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object. Do not include markdown formatting like ```json. Do not include any text before or after the JSON.
                    {{
                      "message": "Your conversational response here...",
                      "scenario": "none",
                      "intensity": "none"
                    }}
                    """
                    
                    try:
                        response = model_ai.generate_content(prompt)
                        text_response = response.text.strip()
                        
                        # Strip away any accidental markdown formatting the AI might add
                        if text_response.startswith("
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1
http://googleusercontent.com/immersive_entry_chip/2

4. Click **"Commit changes"** and refresh your browser.

Now, whether someone asks *"What is 2+2?"* or *"Ban all petrol cars"*, the AI will give a clean conversational response, parse the background scenario flawlessly, and jump straight to the sliders without crashing!
