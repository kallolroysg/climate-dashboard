import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Singapore Carbon Predictor", layout="wide")
st.title("🌱 Singapore 2030 Carbon Emissions & Policy Simulator")

# --- AI SETUP ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.warning("AI API Key not found. Please add it to Streamlit Secrets.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Make sure these names exactly match the files you uploaded to GitHub!
        taxes = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Taxes.csv', skiprows=1)
        subsidies = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Subsidies.csv', skiprows=1)
        return taxes, subsidies
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

taxes_df, subsidies_df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configure Scenario")
scenario = st.sidebar.selectbox(
    "Choose a Policy Scenario:",
    ("Baseline (No Changes)", "Renewable Energy Expansion", "Petrol Car Quota / EV Transition", "High Carbon Tax")
)

# --- FAKE MATH FOR THE CHART ---
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
baseline_emissions = [50, 51, 52, 53, 54, 55, 56, 56.5, 57, 57.5, 58] 

if scenario == "Baseline (No Changes)":
    scenario_emissions = baseline_emissions
elif scenario == "Renewable Energy Expansion":
    scenario_emissions = [50, 51, 52, 53, 54, 53, 51, 48, 45, 42, 38]
elif scenario == "Petrol Car Quota / EV Transition":
    scenario_emissions = [50, 51, 52, 53, 54, 54, 52, 50, 48, 46, 44]
elif scenario == "High Carbon Tax":
    scenario_emissions = [50, 51, 52, 53, 54, 52, 49, 45, 41, 37, 33]

# --- BUILD THE VISUALS ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("CO₂ Emission Projections (Million Tonnes)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=baseline_emissions, mode='lines+markers', name='Baseline', line=dict(color='gray', dash='dash')))
    if scenario != "Baseline (No Changes)":
        fig.add_trace(go.Scatter(x=years, y=scenario_emissions, mode='lines+markers', name=f'Scenario: {scenario}', line=dict(color='#00CC96', width=4)))
    fig.update_layout(xaxis_title="Year", yaxis_title="CO2 Emissions (Mt)", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("OECD Policy Matches")
    st.write(f"Matches for: **{scenario}**")
    
    if scenario == "Renewable Energy Expansion" and not subsidies_df.empty:
        recs = subsidies_df[subsidies_df['Approach'].astype(str).str.contains("Renewable", case=False, na=False)].head(3)
    elif scenario == "Petrol Car Quota / EV Transition" and not subsidies_df.empty:
        recs = subsidies_df[subsidies_df['Approach'].astype(str).str.contains("Vehicle", case=False, na=False)].head(3)
    elif scenario == "High Carbon Tax" and not taxes_df.empty:
        recs = taxes_df[taxes_df['Approach'].astype(str).str.contains("Carbon", case=False, na=False)].head(3)
    else:
        recs = pd.DataFrame()
    
    if not recs.empty:
        st.session_state['policy_context'] = ""
        for idx, row in recs.iterrows():
            with st.expander(f"📍 {row.get('Country', 'Unknown')} - {row.get('Approach', 'Policy')}"):
                st.write(f"**Policy:** {row.get('English name', 'N/A')}")
                st.write(f"**Details:** {row.get('Description', 'No description available.')}")
            st.session_state['policy_context'] += f"Country: {row.get('Country')}, Policy: {row.get('English name')}, Details: {row.get('Description')}\n\n"
    else:
        st.info("Select a policy scenario on the left to see global examples.")

# --- THE AI CHATBOX ---
st.divider()
st.subheader("💬 AI Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask how to adapt these international policies for Singapore..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not API_KEY:
            st.error("Please add your Gemini API Key to Streamlit secrets.")
        else:
            context = st.session_state.get('policy_context', 'No specific policy selected.')
            system_prompt = f"You are a climate policy advisor for Singapore. The user is looking at these OECD policies:\n{context}\nAnswer their question: {prompt}"
            
            try:
                response = model.generate_content(system_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")
