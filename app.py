import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib
import os
import json
import re

# --- PAGE SETUP ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")

# --- STRICT SESSION STATE INITIALIZATION ---
# We exclusively use bracket notation to prevent Streamlit AttributeError crashes
if 'ren_val' not in st.session_state:
    st.session_state['ren_val'] = 0
if 'ev_val' not in st.session_state:
    st.session_state['ev_val'] = 0
if 'tax_val' not in st.session_state:
    st.session_state['tax_val'] = 0
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

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
    
    # --- SIDEBAR: POLICY SCENARIOS ---
    st.sidebar.header("🎛️ Policy Scenarios (Model Inputs)")
    
    # Tied safely to session_state using keys
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
        
        # Safely iterate through the chat history
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
                    prompt = f"""
                    You are a helpful climate policy AI. You do NOT make up mathematical predictions. 
                    The user said: "{user_input}"
                    
                    Task 1: If it's a general question or greeting (e.g., "Hello", "What is 2+2?", "Why is the sky blue?"), answer it conversationally.
                    Task 2: If they propose a policy, explain the policy and state that you are updating the dashboard parameters for the XGBoost model to predict the results. 
                    Task 3: Classify the policy into: "renewable", "ev", or "tax". If it's not a policy, output "none".
                    Task 4: Determine intensity: "low", "medium", or "high". If not a policy, output "none".
                    
                    Output ONLY valid JSON in this exact format, with NO extra text outside the brackets:
                    {{
                      "message": "Your conversational response here...",
                      "scenario": "none/renewable/ev/tax",
                      "intensity": "none/low/medium/high"
                    }}
                    """
                    
                    try:
                        response = model_ai.generate_content(prompt)
                        text_response = response.text
                        
                        # Strict JSON Extraction
                        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                        
                        if json_match:
                            parsed_data = json.loads(json_match.group(0))
                        else:
                            parsed_data = {"message": text_response, "scenario": "none", "intensity": "none"}
                        
                        ai_message = parsed_data.get("message", "I couldn't process that. Could you rephrase?")
                        scenario = parsed_data.get("scenario", "none").lower()
                        intensity = parsed_data.get("intensity", "none").lower()
                        
                        st.markdown(ai_message)
                        st.session_state['chat_history'].append({"role": "assistant", "content": ai_message})
                        
                        # Apply policy changes strictly
                        if scenario != "none" and scenario != "null":
                            val_50_scale = 10 if intensity == "low" else 25 if intensity == "medium" else 50
                            val_20_scale = 5 if intensity == "low" else 10 if intensity == "medium" else 20
                            
                            st.session_state['ren_val'] = 0
                            st.session_state['ev_val'] = 0
                            st.session_state['tax_val'] = 0
                            
                            if "ev" in scenario or "petrol" in scenario:
                                st.session_state['ev_val'] = val_50_scale
                            if "renewable" in scenario or "solar" in scenario:
                                st.session_state['ren_val'] = val_50_scale
                            if "tax" in scenario or "carbon" in scenario:
                                st.session_state['tax_val'] = val_20_scale
                            
                            st.info("🔄 Dashboard sliders automatically updated based on your policy proposal!")
                            st.rerun()

                    except Exception as e:
                        st.error("I couldn't understand that response. Please try again!")
