"""
LivingMemoryOS - Multi-Dataset Unified Streamlit Application
LivingMemoryOS - Streamlit Application Entrypoint
Run with: streamlit run app.py
Allows seamless switching between the new MIMIC-IV 550k Clinical Cohort and the legacy 15k Telemetry dataset.
"""

import streamlit as st
from src.ui.dashboard import render_dashboard

st.set_page_config(
    page_title="LivingMemoryOS | Clinical AI Memory Management",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dataset Selector at the top of the sidebar
with st.sidebar:
    st.markdown("<div style='font-size:1.1rem; font-weight:800; color:#1a76d1; margin-bottom:4px;'>LivingMemoryOS</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; color:#8898aa; text-transform:uppercase; margin-bottom:12px;'>Dataset Environment</div>", unsafe_allow_html=True)
    
    selected_dataset = st.radio(
        "Active Clinical Dataset:",
        [
            "MIMIC-IV Clinical Cohort (550k - New Dataset)",
            "ICU Telemetry Telemetry (15k - Legacy Dataset)",
        ],
        index=0,
        help="Select the clinical dataset version to evaluate.",
    )
    st.markdown("---")

if "MIMIC-IV" in selected_dataset:
    from src.ui.mimic_dashboard import render_mimic_dashboard
    render_mimic_dashboard()
else:
    from src.ui.dashboard import render_dashboard
if __name__ == "__main__":
    render_dashboard()
