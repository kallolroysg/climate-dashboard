import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")
st.title("🇸🇬 Singapore 2030 Carbon ML Simulator")
st.markdown("Real ML predictions based on Our World in Data historical records.")

# --- CONFIGURE AI ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("AI API Key not found.")

# --- LOAD ML MODEL & DATA ---
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('best_co2_model.joblib')
        base_data = pd.read_csv('latest_sg_data.csv')
        return model, base_data
    except Exception as e:
        st.error(f"Please upload the ML model and data to GitHub. Error: {e}")
        return None, None

ml_model, base_data = load_ml_assets()

# --- LOAD OECD DATA ---
@st.cache_data
def load_oecd_data():
    try:
        taxes = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Taxes.csv', skiprows=1)
        subsidies = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Subsidies.csv', skiprows=1)
        return taxes, subsidies
    except:
        return pd.DataFrame(), pd.DataFrame()

taxes_df, subsidies_df = load_oecd_data()

if ml_model is not None and not base_data.empty:
    
    # --- SIDEBAR: POLICY LEVERS ---
    st.sidebar.header("🎛️ Policy Scenarios")
    
    renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Fossil Fuel %)", 0, 50, 0, step=5)
    ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO2 %)", 0, 50, 0, step=5)
    carbon_tax_intensity = st.sidebar.slider("Carbon Tax Intensity (Improve Energy/GDP %)", 0, 20, 0, step=2)

    st.sidebar.divider()
    st.sidebar.subheader("📊 Dataset Overview")
    st.sidebar.caption("- Source: Our World in Data")
    st.sidebar.caption("- Target: CO2 Emissions")
    st.sidebar.caption("- Model: Best performing Regressor (XGB/RF)")

    # --- ML PREDICTION LOGIC ---
    # 1. Baseline Prediction (No Changes)
    features = ['year', 'population', 'gdp', 'primary_energy_consumption', 'energy_per_gdp', 'energy_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'fossil_fuel_co2']
    
    base_input = base_data.copy()
    base_input['year'] = 2030 # Projecting to 2030
    baseline_pred = ml_model.predict(base_input[features])[0]

    # 2. Adjusted Prediction (Applying sliders to variables as teammate requested)
    adj_input = base_input.copy()
    
    # EV Transition reduces oil_co2
    adj_input['oil_co2'] = adj_input['oil_co2'] * (1 - (ev_intensity/100))
    # Renewable reduces total fossil fuel proxy
    adj_input['fossil_fuel_co2'] = adj_input['fossil_fuel_co2'] * (1 - (renewable_intensity/100))
    # Carbon tax improves energy intensity
    adj_input['energy_per_gdp'] = adj_input['energy_per_gdp'] * (1 - (carbon_tax_intensity/100))
    
    adjusted_pred = ml_model.predict(adj_input[features])[0]
    total_reduction = baseline_pred - adjusted_pred

    # --- DASHBOARD METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline 2030 Prediction", f"{baseline_pred:.2f} Mt")
    col2.metric("Policy-Adjusted Prediction", f"{adjusted_pred:.2f} Mt", f"-{total_reduction:.2f} Mt", delta_color="inverse")
    col3.metric("Reduction", f"{(total_reduction/baseline_pred)*100:.1f}%")

    st.write("---")

    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📈 ML Forecast", "📋 OECD Policies", "🤖 AI Classifier"])

    with tab1:
        st.subheader("Model Projections vs Baseline")
        # Creating a simple chart with historical + predicted data
        hist_years = [2020, 2021, 2022] # Mock recent history for visual continuity
        hist_co2 = [49.0, 50.5, 51.5]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_years + [2030], y=hist_co2 + [baseline_pred], mode='lines+markers', name='Baseline (No Action)', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=hist_years + [2030], y=hist_co2 + [adjusted_pred], mode='lines+markers', name='Policy Interventions', line=dict(color='green', width=4)))
        
        fig.update_layout(xaxis=dict(type='category'), yaxis_title="CO2 (Mt)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Dynamic recommendations based on active policies:")
        if ev_intensity > 0:
            st.info("EV Transition active. Matching OECD Policies:")
            st.dataframe(subsidies_df[subsidies_df['Approach'].astype(str).str.contains("Vehicle", case=False, na=False)][['Country', 'English name']].head(3))
        if carbon_tax_intensity > 0:
            st.info("Carbon Tax active. Matching OECD Policies:")
            st.dataframe(taxes_df[taxes_df['Approach'].astype(str).str.contains("Carbon", case=False, na=False)][['Country', 'English name']].head(3))

    with tab3:
        st.subheader("AI Policy Classifier")
        st.caption("As per project requirements, this AI does NOT invent numbers. It only classifies policy text into our scenario JSON format.")
        
        user_policy = st.text_area("Enter a proposed climate policy for classification:")
        if st.button("Classify Policy"):
            if not API_KEY:
                st.error("API Key missing.")
            else:
                prompt = f"""
                You are a strictly constrained data classifier. Do NOT generate predictions or make up numbers.
                Classify the user's policy into one of these scenarios: "Renewable energy expansion", "Petrol car quota / EV transition", or "Carbon tax".
                Output ONLY valid JSON format like this example:
                {{
                  "scenario": "Petrol car quota / EV transition",
                  "intensity": "medium",
                  "affected_variables": ["oil_co2", "fossil_fuel_co2"]
                }}
                
                User Policy: {user_policy}
                """
                response = model_ai.generate_content(prompt)
                st.json(response.text.replace("```json","").replace("```",""))
