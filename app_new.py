import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="LivingMemoryOS",
    layout="wide"
)

# =========================
# STYLE
# =========================

st.markdown("""
<style>

.stApp{
    background-color:#FFDDEB;
}

h1,h2,h3{
    color:black;
}

[data-testid="stMetricValue"]{
    color:black;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

results = pd.read_csv(
    "livingmemory_results_v4.csv"
)

capacity = pd.read_csv(
    "capacity_evolution_v4.csv"
)

# =========================
# HEADER
# =========================

st.title(
    "🧠 LivingMemoryOS"
)

st.subheader(
    "Self-Evolving Clinical Memory Architecture"
)

st.write(
"""
AI-driven memory retention for healthcare edge devices.

Patients are prioritized according to:
- AI Risk
- Biomarker Risk
- Care Escalation
- Patient Criticality Inheritance
"""
)

# =========================
# METRICS
# =========================

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Retained Pages",
    len(results)
)

col2.metric(
    "Protected Pages",
    results["protected"].sum()
)

col3.metric(
    "Average CMS",
    round(results["cms"].mean(),3)
)

col4.metric(
    "Adaptive Capacity",
    int(capacity["adaptive_capacity"][0])
)

st.divider()

# =========================
# CMS DISTRIBUTION
# =========================

st.subheader(
    "Clinical Memory Score Distribution"
)

fig = px.histogram(
    results,
    x="cms"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# TOP PATIENTS
# =========================

st.subheader(
    "Highest Priority Patients"
)

top = results.sort_values(
    "cms",
    ascending=False
).head(20)

st.dataframe(top)

# =========================
# CARE RISK
# =========================

st.subheader(
    "AI Risk vs CMS"
)

fig2 = px.scatter(
    results,
    x="ai_risk",
    y="cms"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================
# CAPACITY
# =========================

st.subheader(
    "Memory Capacity Evolution"
)

st.dataframe(capacity)

st.success(
    "LivingMemoryOS Simulation Complete"
)