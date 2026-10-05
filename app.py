"""
LivingMemoryOS is a clinical memory-management and patient-prioritization framework
that processes healthcare datasets, generates mortality-risk predictions using
machine learning, computes Clinical Memory Scores (CMS), applies prognostic
retention policies, and simulates adaptive memory replacement strategies. The
system supports both the MIMIC-IV clinical cohort and a legacy ICU telemetry
dataset, providing patient triage, tier assignment, memory retention analysis,
capacity adaptation, and comparative benchmarking against standard FIFO memory
management approaches.
"""

import streamlit as st

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
            "ICU Telemetry (15k - Legacy Dataset)",
        ],
        index=0,
        help="Select the clinical dataset version to evaluate.",
    )
    st.markdown("---")

import importlib

if "MIMIC-IV" in selected_dataset:
    import src.ui.dashboard_new as d_mod
    importlib.reload(d_mod)
    render_dashboard = d_mod.render_dashboard
else:
    import src.ui.dashboard as d_mod
    importlib.reload(d_mod)
    render_dashboard = d_mod.render_dashboard

if __name__ == "__main__":
    render_dashboard()