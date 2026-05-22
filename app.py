import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib
import os
import json
import re
import numpy as np

# --- PAGE SETUP & SESSION STATE ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")

if 'ren_val' not in st.session_state: st.session_state['ren_val'] = 0
if 'ev_val' not in st.session_state: st.session_state['ev_val'] = 0
if 'tax_val' not in st.session_state: st.session_state['tax_val'] = 0
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("Powered by Our World in Data, XGBoost, and **Gemini 3.5 Flash**.")

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
features_list = ['year', 'population', 'gdp', 'primary_energy_consumption', 'energy_per_gdp', 'energy_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'fossil_fuel_co2']

# --- MAIN DASHBOARD ---
if ml_model is not None and full_data is not None:
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🎛️ Policy Scenarios")
    
    renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Fossil Fuel %)", 0, 50, value=st.session_state['ren_val'], step=5)
    ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO₂ %)", 0, 50, value=st.session_state['ev_val'], step=5)
    carbon_tax_intensity = st.sidebar.slider("Carbon Tax (Improve Energy/GDP %)", 0, 20, value=st.session_state['tax_val'], step=2)

    st.session_state['ren_val'] = renewable_intensity
    st.session_state['ev_val'] = ev_intensity
    st.session_state['tax_val'] = carbon_tax_intensity

    st.sidebar.divider()
    
    # --- A+ TRANSPARENCY FEATURES ---
    st.sidebar.subheader("📊 Model & Data Transparency")
    st.sidebar.caption("✅ Source: Our World in Data (World Bank & GCP)")
    st.sidebar.caption("✅ Model: XGBoost Regressor")
    
    # 1. Feature Importance Chart
    with st.sidebar.expander("🧠 Model Explainability (Feature Weights)"):
        st.caption("This chart shows which variables the XGBoost algorithm found most mathematically impactful for predicting Singapore's CO₂ emissions.")
        if hasattr(ml_model, 'feature_importances_'):
            importances = ml_model.feature_importances_ * 100
            imp_df = pd.DataFrame({'Feature': features_list, 'Importance': importances}).sort_values(by='Importance', ascending=True)
            
            fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker_color='#2ca02c'))
            fig_imp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, xaxis_title="Impact (%)", font=dict(size=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    # 2. Raw Data Viewer (FIXED: Removed duplicate 'year' column)
    with st.sidebar.expander("🔍 View Training Dataset"):
        st.caption("Raw historical data for Singapore, cleaned and utilized for ML training.")
        st.dataframe(full_data[['co2'] + features_list].tail(20), height=200)

    # --- ML PREDICTION ENGINE ---
    hist_data = full_data.tail(10)
    hist_years = hist_data['year'].tolist()
    hist_co2 = hist_data['co2'].tolist()
    
    latest_features = full_data.tail(1).iloc[0]
    latest_year = int(latest_features['year'])
    
    future_years = list(range(latest_year + 1, 2036))
    baseline_preds = []
    policy_preds = []

    for year in future_years:
        years_ahead = year - latest_year
        economic_cycle = 1.0 + (np.sin((year - latest_year) * 0.8) * 0.01) 
        
        base_row = latest_features.copy()
        base_row['year'] = year
        base_row['gdp'] = base_row['gdp'] * ((1.02) ** years_ahead) * economic_cycle
        base_row['population'] = base_row['population'] * ((1.008) ** years_ahead)
        base_row['primary_energy_consumption'] = base_row['primary_energy_consumption'] * ((1.012) ** years_ahead) * economic_cycle
        base_row['oil_co2'] = base_row['oil_co2'] * ((1.01) ** years_ahead) * economic_cycle
        base_row['fossil_fuel_co2'] = base_row['fossil_fuel_co2'] * ((1.01) ** years_ahead) * economic_cycle

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

    idx_2030 = future_years.index(2030) if 2030 in future_years else -1
    pred_2030_base = baseline_preds[idx_2030]
    pred_2030_policy = policy_preds[idx_2030]
    reduction = pred_2030_base - pred_2030_policy

    st.write("---")

    # --- SINGLE PAGE LAYOUT ---
    col_chart, col_chat = st.columns([1.5, 1])

    with col_chart:
        met1, met2, met3 = st.columns(3)
        met1.metric("Baseline 2030 Forecast", f"{pred_2030_base:.2f} Mt")
        met2.metric("Policy-Adjusted 2030", f"{pred_2030_policy:.2f} Mt", f"-{reduction:.2f} Mt", delta_color="inverse")
        if pred_2030_base > 0:
            met3.metric("Reduction (%)", f"{(reduction/pred_2030_base)*100:.1f}%")

        st.subheader("📈 ML Forecast vs Scenarios")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_years, y=hist_co2, mode='lines+markers', name='Historical Data', line=dict(color='black', width=2)))
        
        connect_year = [hist_years[-1], future_years[0]]
        connect_base_co2 = [hist_co2[-1], baseline_preds[0]]
        connect_pol_co2 = [hist_co2[-1], policy_preds[0]]

        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_base_co2 + baseline_preds[1:], mode='lines', name='Baseline Forecast', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_pol_co2 + policy_preds[1:], mode='lines', name='Policy Intervention', line=dict(color='green', width=3)))
        
        fig.update_layout(xaxis_title="Year", yaxis_title="CO₂ Emissions (Mt)", height=450, hovermode="x unified", xaxis=dict(type='category'), margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.info("ℹ️ **Methodology Note:** These projections represent *scenario simulations* where input variables (e.g., oil CO₂) are adjusted prior to prediction. The underlying model predicts outcomes based on historical patterns in the dataset, not hallucinatory guessing.")

    with col_chat:
        st.subheader("🤖 Smart AI Controller")
        st.caption("Ask anything! Or propose a policy to watch the chart update in real-time.")
        
        chat_container = st.container(height=550)
        
        with chat_container:
            for msg in st.session_state['chat_history']:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        if user_input := st.chat_input("E.g., What if we ban all petrol cars?"):
            
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                
            with chat_container:
                with st.chat_message("assistant"):
                    if not API_KEY:
                        st.error("API Key missing.")
                    else:
                        prompt = f"""
                        You are a helpful climate policy AI advising the Singapore Government. You do NOT make up mathematical predictions. 
                        The user said: "{user_input}"
                        
                        Task 1: If it's a general/dumb question (e.g., "Hello", "What is 2+2?"), answer it simply and conversationally in the "message" field.
                        Task 2: If they propose a policy, write a short conversational response in the "message" field. 
                        - FIRST, provide 1-2 bullet points on how this could be implemented LOCALLY in Singapore. 
                        - THEN, provide 1-2 bullet points of successful OVERSEAS examples.
                        - FINALLY, state clearly at the end: "I am updating the dashboard parameters..."
                        Task 3: Classify the policy into: "renewable", "ev", or "tax". If not a policy, output "none".
                        Task 4: Determine intensity: "low", "medium", or "high". If not a policy, output "none".
                        
                        CRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object.
                        {{
                          "message": "Your conversational response here...",
                          "scenario": "none",
                          "intensity": "none"
                        }}
                        """
                        
                        try:
                            response = model_ai.generate_content(prompt)
                            text_response = response.text
                            
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
                            
                            if scenario != "none" and scenario != "null":
                                val_50_scale = 10 if intensity == "low" else 20 if intensity == "medium" else 35
                                val_20_scale = 4 if intensity == "low" else 8 if intensity == "medium" else 15
                                
                                st.session_state['ren_val'] = 0
                                st.session_state['ev_val'] = 0
                                st.session_state['tax_val'] = 0
                                
                                if "ev" in scenario or "petrol" in scenario:
                                    st.session_state['ev_val'] = val_50_scale
                                if "renewable" in scenario or "solar" in scenario:
                                    st.session_state['ren_val'] = val_50_scale
                                if "tax" in scenario or "carbon" in scenario:
                                    st.session_state['tax_val'] = val_20_scale
                                
                                st.rerun()

                        except Exception as e:
                            st.error(f"Oops! The AI didn't format its response correctly. Please try again! (Debug: {e})")
