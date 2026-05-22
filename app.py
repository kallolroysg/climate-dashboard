import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="SG Climate ML Simulator", page_icon="🇸🇬", layout="wide")
st.title("🇸🇬 Singapore CO₂ Machine Learning Simulator")
st.markdown("Powered by Our World in Data, XGBoost, and Google Gemini.")

# --- CONFIGURE AI ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.warning("AI API Key not found.")

# --- LOAD ML MODEL & DATA ---
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('best_co2_model.joblib')
        # STRICT FIX: Using the exact file name you downloaded from Colab!
        full_data = pd.read_csv('cleaned_sg_co2_data.csv')
        return model, full_data
    except Exception as e:
        st.error(f"⚠️ Error loading files: {e}. Please ensure 'best_co2_model.joblib' and 'cleaned_sg_co2_data.csv' are uploaded to GitHub.")
        return None, None

ml_model, full_data = load_ml_assets()

if ml_model is not None and full_data is not None:
    
    # --- SIDEBAR: POLICY SCENARIOS ---
    st.sidebar.header("🎛️ Policy Scenarios (Model Inputs)")
    
    renewable_intensity = st.sidebar.slider("Renewable Expansion (Reduce Fossil Fuel %)", 0, 50, 0, step=5)
    ev_intensity = st.sidebar.slider("EV Transition (Reduce Oil CO₂ %)", 0, 50, 0, step=5)
    carbon_tax_intensity = st.sidebar.slider("Carbon Tax (Improve Energy/GDP %)", 0, 20, 0, step=2)

    st.sidebar.divider()
    st.sidebar.subheader("📊 Dataset & Model Overview")
    st.sidebar.caption("✅ Source: Our World in Data")
    st.sidebar.caption("✅ Target: CO2 Emissions")
    st.sidebar.caption("✅ Model: XGBoost Regressor")
    st.sidebar.caption(f"✅ Training Records: {len(full_data)}")

    # --- ML PREDICTION ENGINE (PROMPTS 4 & 5) ---
    features_list = ['year', 'population', 'gdp', 'primary_energy_consumption', 'energy_per_gdp', 'energy_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'fossil_fuel_co2']
    
    # Get historical data for the chart (last 10 years available)
    hist_data = full_data.tail(10)
    hist_years = hist_data['year'].tolist()
    hist_co2 = hist_data['co2'].tolist()
    
    # Grab the most recent year's data to use as the baseline for the future
    latest_features = full_data.tail(1).iloc[0]
    latest_year = int(latest_features['year'])
    
    # Forecast from the latest data point out to 2035
    future_years = list(range(latest_year + 1, 2036))
    baseline_preds = []
    policy_preds = []

    for year in future_years:
        # 1. Baseline Row (No policy changes, just updating the year)
        base_row = latest_features.copy()
        base_row['year'] = year
        base_df = pd.DataFrame([base_row])[features_list]
        b_pred = ml_model.predict(base_df)[0]
        baseline_preds.append(b_pred)

        # 2. Policy-Adjusted Row (Applying the exact logic from Prompt 5)
        pol_row = base_row.copy()
        pol_row['oil_co2'] = pol_row['oil_co2'] * (1 - (ev_intensity/100))
        pol_row['fossil_fuel_co2'] = pol_row['fossil_fuel_co2'] * (1 - (renewable_intensity/100))
        pol_row['energy_per_gdp'] = pol_row['energy_per_gdp'] * (1 - (carbon_tax_intensity/100))
        
        pol_df = pd.DataFrame([pol_row])[features_list]
        p_pred = ml_model.predict(pol_df)[0]
        policy_preds.append(p_pred)

    # --- DASHBOARD METRICS ---
    # Find the index for 2030 to display in the top metrics
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
    tab1, tab2 = st.tabs(["📈 ML Forecast vs Scenarios", "🤖 Strict AI Classifier"])

    with tab1:
        st.subheader("Predictive ML Forecast (Historical + Future to 2035)")
        
        fig = go.Figure()
        # Historical Line
        fig.add_trace(go.Scatter(x=hist_years, y=hist_co2, mode='lines+markers', name='Historical Data', line=dict(color='black', width=2)))
        
        # Connect history to forecast smoothly
        connect_year = [hist_years[-1], future_years[0]]
        connect_base_co2 = [hist_co2[-1], baseline_preds[0]]
        connect_pol_co2 = [hist_co2[-1], policy_preds[0]]

        # Baseline Forecast Line
        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_base_co2 + baseline_preds[1:], mode='lines', name='Baseline Forecast', line=dict(color='gray', dash='dash')))
        
        # Policy Forecast Line
        fig.add_trace(go.Scatter(x=connect_year + future_years[1:], y=connect_pol_co2 + policy_preds[1:], mode='lines', name='Policy Intervention', line=dict(color='green', width=3)))
        
        # Format the X-axis to display whole years as text to prevent decimal errors
        fig.update_layout(xaxis_title="Year", yaxis_title="CO₂ Emissions (Mt)", height=450, hovermode="x unified", xaxis=dict(type='category'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Strict Policy Classifier (Prompt 6)")
        st.caption("As requested, this AI does NOT generate mathematical predictions. It only classifies text into structured JSON.")
        
        user_policy = st.text_area("Enter a proposed climate policy for Singapore:")
        if st.button("Classify Policy"):
            if not API_KEY:
                st.error("API Key missing in secrets.")
            else:
                prompt = f"""
                You are a strictly constrained data classifier. Do NOT make up numbers or predict CO2.
                Classify the user's policy into one of these scenarios: "Renewable energy expansion", "Petrol car quota / EV transition", or "Carbon tax".
                Output ONLY valid JSON format like this example:
                {{
                  "scenario": "Petrol car quota / EV transition",
                  "intensity": "medium",
                  "affected_variables": ["oil_co2", "fossil_fuel_co2"]
                }}
                
                User Policy: {user_policy}
                """
                try:
                    response = model_ai.generate_content(prompt)
                    st.json(response.text.replace("```json","").replace("```",""))
                except Exception as e:
                    st.error(f"AI Error: {e}")
