"""
LivingMemoryOS v4 - MIMIC-IV Clinical Cohort Web Application
Run with: streamlit run app_new.py
Dedicated frontend for the 550,818 inpatient admissions dataset.
"""

import streamlit as st
from src.ui.dashboard_new import render_dashboard

if __name__ == "__main__":
    render_dashboard()
