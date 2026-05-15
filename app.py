import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai

# --- 1. PROFESSIONAL PAGE SETUP ---
st.set_page_config(page_title="SG Climate Simulator", page_icon="🇸🇬", layout="wide")

# Custom CSS to make it look like a professional web app
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    h1 { color: #0F172A; font-weight: 700; }
    .stMetric { background-color: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

st.title("🇸🇬 Singapore 2030 Carbon Policy Simulator")
st.markdown("Adjust the policy levers on the left to simulate carbon reductions and view OECD recommendations.")

# --- 2. CONFIGURE FREE AI ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash') # Using the updated model!
else:
    st.error("AI API Key not found. Please add it to Streamlit Secrets.")

# --- 3. LOAD OECD DATA ---
@st.cache_data
def load_data():
    try:
        taxes = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Taxes.csv', skiprows=1)
        subsidies = pd.read_csv('IFCMA_ClimatePolicyDashboard_Data_April 2026.xlsx - Subsidies.csv', skiprows=1)
        return taxes, subsidies
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

taxes_df, subsidies_df = load_data()

# --- 4. INTERACTIVE SIDEBAR LEVERS (The Upgrade) ---
st.sidebar.header("🎛️ Policy Levers")
st.sidebar.markdown("Adjust targets to see the impact on 2030 emissions.")

# Singapore realistic starting points
renewable_target = st.sidebar.slider("Renewable Energy Share (%)", min_value=5, max_value=50, value=5, step=1)
ev_adoption = st.sidebar.slider("EV Adoption Rate (%)", min_value=10, max_value=100, value=15, step=5)
carbon_tax = st.sidebar.slider("Carbon Tax (S$/tonne)", min_value=25, max_value=150, value=45, step=5)

# --- 5. DYNAMIC PREDICTION MATH ---
# Base 2030 emissions without new policies is ~58 Mt
base_2030 = 58.0 

# Calculate reductions based on slider movements (Simple assumptions for the model)
renewable_reduction = (renewable_target - 5) * 0.15  # Every 1% above base reduces 0.15 Mt
ev_reduction = (ev_adoption - 15) * 0.08             # Every 1% above base reduces 0.08 Mt
tax_reduction = (carbon_tax - 45) * 0.05             # Every $1 above base reduces 0.05 Mt

total_reduction = renewable_reduction + ev_reduction + tax_reduction
projected_2030 = base_2030 - total_reduction

# Generate the data points for the chart
years = list(range(2024, 2031))
baseline_curve = [54.0, 55.0, 56.0, 56.5, 57.0, 57.5, 58.0]
# Calculate a smooth glide path to the new 2030 target
scenario_curve = [54.0]
for i in range(1, 7):
    scenario_curve.append(54.0 + (projected_2030 - 54.0) * (i / 6))

# --- 6. TOP METRICS DASHBOARD ---
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(label="Projected 2030 Emissions", value=f"{projected_2030:.1f} Mt", delta=f"-{total_reduction:.1f} Mt vs Baseline", delta_color="inverse")
col_m2.metric(label="Renewable Grid Target", value=f"{renewable_target}%", delta="Solar Expansion")
col_m3.metric(label="Carbon Price", value=f"S${carbon_tax}", delta="Cost to Emitters", delta_color="off")
st.write("") # Spacer

# --- 7. MAIN LAYOUT (Chart & Policies) ---
col1, col2 = st.columns([5, 3]) # Makes the chart wider than the policy text

with col1:
    # Beautiful Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=baseline_curve, mode='lines+markers', name='Business as Usual', line=dict(color='#94A3B8', dash='dash', width=3)))
    fig.add_trace(go.Scatter(x=years, y=scenario_curve, mode='lines+markers+text', name='Policy Intervention', 
                             line=dict(color='#10B981', width=4),
                             text=["", "", "", "", "", "", f"{projected_2030:.1f} Mt"], textposition="bottom center"))
    
    fig.update_layout(
        title="CO₂ Emissions Trajectory (2024 - 2030)",
        xaxis_title="Year", 
        yaxis_title="Million Tonnes (Mt) CO₂",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Policy Matches (OECD)")
    st.write("Dynamic recommendations based on your highest slider setting:")
    
    # Logic to show the most relevant policy based on the user's sliders
    st.session_state['policy_context'] = ""
    recs = pd.DataFrame()
    
    if carbon_tax >= 80 and not taxes_df.empty:
        st.info("💡 High Carbon Tax detected. Showing global tax policies:")
        recs = taxes_df[taxes_df['Approach'].astype(str).str.contains("Carbon", case=False, na=False)].head(3)
    elif renewable_target >= 30 and not subsidies_df.empty:
        st.info("💡 High Renewable target detected. Showing clean energy subsidies:")
        recs = subsidies_df[subsidies_df['Approach'].astype(str).str.contains("Renewable", case=False, na=False)].head(3)
    elif ev_adoption >= 50 and not subsidies_df.empty:
        st.info("💡 High EV Adoption detected. Showing vehicle subsidies:")
        recs = subsidies_df[subsidies_df['Approach'].astype(str).str.contains("Vehicle", case=False, na=False)].head(3)
    else:
        st.warning("Increase the sliders on the left to trigger aggressive policy recommendations.")
        
    if not recs.empty:
        for idx, row in recs.iterrows():
            with st.expander(f"📍 {row.get('Country', 'Unknown')} - {row.get('Approach', 'Policy')}"):
                st.write(f"**Policy:** {row.get('English name', 'N/A')}")
                st.write(f"**Details:** {row.get('Description', 'No description available.')}")
            st.session_state['policy_context'] += f"Country: {row.get('Country')}, Policy: {row.get('English name')}, Details: {row.get('Description')}\n\n"

# --- 8. AI CHAT INTERFACE ---
st.divider()
st.subheader("💬 AI Policy Advisor")
st.write(f"Ask the AI how to implement the policies above to reach your **{projected_2030:.1f} Mt** target.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., How can Singapore fund these EV subsidies?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not API_KEY:
            st.error("Please add your Gemini API Key to Streamlit secrets.")
        else:
            context = st.session_state.get('policy_context', 'No specific policy selected.')
            system_prompt = f"""You are a climate policy advisor for Singapore. 
            The user has set sliders resulting in a 2030 target of {projected_2030:.1f} Mt.
            OECD context:\n{context}\n
            Answer concisely and professionally. Question: {prompt}"""
            
            try:
                response = model.generate_content(system_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
