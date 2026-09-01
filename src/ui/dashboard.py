"""
Streamlit Web Dashboard for LivingMemoryOS
Inspired by clean Mediplus Medical Clinical UI
Pure White Base, Royal Blue & Coral Red Accents, Zero Emojis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import (
    DEFAULT_TIER_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_CARE_LEVEL_THRESHOLDS,
    FEATURES,
)
from src.models.classifier import MortalityClassifier
from src.memory.simulator import LivingMemorySimulator


# Define cache function at top-level to prevent Python inspect tokenization errors
@st.cache_resource(show_spinner="Initializing Clinical Machine Learning Model...")
def get_simulator():
    classifier = MortalityClassifier()
    classifier.load_and_train()
    return LivingMemorySimulator(classifier)


def render_dashboard():
    # Page configuration
    st.set_page_config(
        page_title="LivingMemoryOS | Clinical AI Memory Management",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Mediplus-Inspired Medical Light Theme CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        /* Base Reset */
        .stApp, html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #ffffff !important;
            font-family: 'Poppins', 'Inter', -apple-system, sans-serif !important;
            color: #2c2d3f !important;
        }

        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1250px !important;
        }

        /* Sidebar - Clean Medical White & Soft Grey */
        section[data-testid="stSidebar"] {
            background-color: #f8fbfe !important;
            border-right: 1px solid #e1effa !important;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #f8fbfe !important;
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .anim-fade {
            animation: fadeInUp 0.4s ease-out forwards;
        }

        /* Top Brand Navigation Bar */
        .mediplus-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 20px;
            background: #ffffff;
            border-bottom: 1px solid #edf2f7;
            margin-bottom: 20px;
        }

        .brand-logo {
            font-size: 1.6rem;
            font-weight: 800;
            color: #2c2d3f;
            letter-spacing: -0.5px;
        }

        .brand-logo span {
            color: #1a76d1;
        }

        .brand-tagline {
            font-size: 0.8rem;
            color: #8898aa;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Hero Banner */
        .mediplus-hero {
            background: linear-gradient(135deg, #f4f9fd 0%, #ffffff 100%);
            border: 1px solid #e1effa;
            border-radius: 12px;
            padding: 32px 28px;
            text-align: center;
            margin-bottom: 24px;
            position: relative;
        }

        .hero-pretitle {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #1a76d1;
            margin-bottom: 8px;
        }

        .hero-main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2c2d3f;
            line-height: 1.25;
            margin: 0 0 10px 0;
        }

        .hero-main-title span {
            color: #1a76d1;
        }

        .hero-description {
            font-size: 0.95rem;
            color: #64748b;
            max-width: 760px;
            margin: 0 auto 18px auto;
            line-height: 1.6;
        }

        .pill-badge {
            display: inline-block;
            background: #e8f3fc;
            color: #1a76d1;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 4px 14px;
            border-radius: 20px;
            border: 1px solid #d0e7f9;
        }

        /* Stats Royal Blue Banner (Mediplus Counter Section) */
        .stats-banner {
            background: linear-gradient(135deg, #1a76d1 0%, #135da7 100%);
            border-radius: 12px;
            padding: 24px 20px;
            margin-bottom: 28px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(26, 118, 209, 0.18);
        }

        .stat-box {
            text-align: center;
            padding: 10px 8px;
            border-right: 1px solid rgba(255, 255, 255, 0.18);
        }

        .stat-box:last-child {
            border-right: none;
        }

        .stat-number {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
            margin-bottom: 4px;
        }

        .stat-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #e1effa;
        }

        .stat-sub {
            font-size: 0.72rem;
            color: #b9dcfa;
            margin-top: 3px;
        }

        /* Mediplus Feature & Container Cards */
        .feature-card {
            background: #ffffff;
            border: 1px solid #edf2f7;
            border-radius: 10px;
            padding: 22px 18px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
            transition: all 0.25s ease;
            height: 100%;
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(26, 118, 209, 0.08);
            border-color: #d0e7f9;
        }

        .card-header-line {
            font-size: 1.05rem;
            font-weight: 700;
            color: #2c2d3f;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f1f5f9;
        }

        /* Badges */
        .badge-emergency-coral {
            background-color: #fff0ed;
            color: #fe7660;
            border: 1px solid #ffdcd6;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
        }

        .badge-high-blue {
            background-color: #e8f3fc;
            color: #1a76d1;
            border: 1px solid #c9e4fb;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
        }

        .badge-normal-teal {
            background-color: #edfdf8;
            color: #0d9488;
            border: 1px solid #c2f5e9;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
        }

        /* Streamlit Form Styling */
        div[data-testid="stForm"] {
            background-color: #ffffff !important;
            border: 1px solid #e1effa !important;
            border-radius: 12px !important;
            padding: 24px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
        }

        /* Tab styling */
        button[data-baseweb="tab"] {
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            padding: 10px 20px !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #1a76d1 !important;
            border-bottom-color: #1a76d1 !important;
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border: 1px solid #edf2f7;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Brand Navbar
    st.markdown(
        """
        <div class="mediplus-navbar">
            <div class="brand-logo">Living<span>MemoryOS</span></div>
            <div class="brand-tagline">Clinical Telemetry Memory Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        """
        <div class="mediplus-hero">
            <div class="hero-pretitle">Clinical-Aware Memory Replacement</div>
            <h1 class="hero-main-title">Intelligent Memory Allocation for <span>ICU Telemetry</span></h1>
            <p class="hero-description">
                Preserving high-risk physiological data in resource-constrained medical edge hardware using calibrated machine learning risk triage and non-evictable min-heap tier budgeting.
            </p>
            <div>
                <span class="pill-badge">Calibrated Isotonic Model</span>
                <span class="pill-badge" style="margin-left:6px;">O(log n) Min-Heap Replacement</span>
                <span class="pill-badge" style="margin-left:6px;">Zero Emergency Displacement</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieve simulator
    try:
        simulator = get_simulator()
    except Exception as e:
        st.error(f"Initialization failure: {e}")
        st.stop()

    # Sidebar: Clean Control Panel
    with st.sidebar:
        st.markdown("<div style='font-size:1.15rem; font-weight:700; color:#2c2d3f;'>System Controls</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#8898aa; margin-bottom:16px;'>Physical Memory Budget Configuration</p>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#1a76d1; text-transform:uppercase; margin-bottom:8px;'>Memory Tier Slots</div>", unsafe_allow_html=True)
        cap_emergency = st.slider("Emergency Tier (Protected)", 5, 50, DEFAULT_TIER_CAPACITY["EMERGENCY"])
        cap_high = st.slider("High Priority Tier", 10, 80, DEFAULT_TIER_CAPACITY["HIGH_PRIORITY"])
        cap_normal = st.slider("Normal Telemetry Tier", 10, 100, DEFAULT_TIER_CAPACITY["NORMAL"])

        tier_capacities = {
            "EMERGENCY": cap_emergency,
            "HIGH_PRIORITY": cap_high,
            "NORMAL": cap_normal,
        }
        total_slots = sum(tier_capacities.values())

        st.markdown(
            f"""
            <div style="background:#e8f3fc; border:1px solid #c9e4fb; border-radius:8px; padding:12px; margin:16px 0; text-align:center;">
                <div style="font-size:0.72rem; font-weight:700; color:#1a76d1; text-transform:uppercase; letter-spacing:0.5px;">Total Hardware Budget</div>
                <div style="font-size:1.45rem; font-weight:800; color:#2c2d3f;">{total_slots} Pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#1a76d1; text-transform:uppercase; margin:16px 0 8px 0;'>Risk Threshold</div>", unsafe_allow_html=True)
        critical_threshold = st.slider(
            "Critical Cutoff Boundary",
            min_value=0.10,
            max_value=0.60,
            value=DEFAULT_CRITICAL_THRESHOLD,
            step=0.01,
        )

        with st.expander("CMS Clinical Weights"):
            w_crit = st.number_input("Mortality Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["criticality"], 0.05)
            w_sev = st.number_input("Acuity Severity Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["severity"], 0.05)
            w_bed = st.number_input("Bed Priority Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["bed_priority"], 0.05)
            w_bio = st.number_input("Lactate Biomarker Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["biomarker_risk"], 0.05)
            w_sep = st.number_input("Sepsis Flag Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["sepsis_risk"], 0.05)
            w_esc = st.number_input("Escalation Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["escalation_score"], 0.05)

            cms_weights = {
                "criticality": w_crit,
                "severity": w_sev,
                "bed_priority": w_bed,
                "biomarker_risk": w_bio,
                "sepsis_risk": w_sep,
                "escalation_score": w_esc,
            }

        st.markdown("---")
        st.markdown("<div style='font-size:0.75rem; color:#8898aa; text-align:center;'>Mediplus Medical Clinical Palette</div>", unsafe_allow_html=True)

    # Run Simulation
    sim_results = simulator.run_simulation(
        tier_capacities=tier_capacities,
        cms_weights=cms_weights,
        critical_threshold=critical_threshold,
    )

    # Mediplus Stats Banner (Royal Blue)
    st.markdown(
        f"""
        <div class="stats-banner anim-fade">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr);">
                <div class="stat-box">
                    <div class="stat-number">{sim_results['living_critical_pages']}</div>
                    <div class="stat-label">LivingMemory Retained</div>
                    <div class="stat-sub">out of {total_slots} total page slots</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sim_results['fifo_critical_pages']}</div>
                    <div class="stat-label">FIFO Baseline Retained</div>
                    <div class="stat-sub">unaware temporal buffer</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">+{sim_results['improvement_pct']}%</div>
                    <div class="stat-label">Retention Improvement</div>
                    <div class="stat-sub">critical telemetry preservation</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sim_results['rejected_emergency']}</div>
                    <div class="stat-label">Emergency Rejections</div>
                    <div class="stat-sub">capacity overflow alerts</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "Memory Simulation & Comparative Analytics",
        "Real-Time Patient Telemetry Triage",
        "Clinical ML Model Validation",
    ])

    # ---------------------------------------------------------
    # TAB 1: MEMORY SIMULATION & ANALYTICS
    # ---------------------------------------------------------
    with tab1:
        st.markdown('<div class="anim-fade">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="card-header-line">Critical Telemetry Retention Comparison</div>
                """,
                unsafe_allow_html=True,
            )
            comp_df = pd.DataFrame({
                "Algorithm": ["FIFO Baseline", "LivingMemoryOS (CAMR)"],
                "Critical Telemetry Retained": [sim_results["fifo_critical_pages"], sim_results["living_critical_pages"]],
                "Standard Telemetry Retained": [
                    len(sim_results["fifo_memory"]) - sim_results["fifo_critical_pages"],
                    len(sim_results["living_memory"]) - sim_results["living_critical_pages"],
                ],
            })
            fig_bar = px.bar(
                comp_df,
                x="Algorithm",
                y=["Critical Telemetry Retained", "Standard Telemetry Retained"],
                barmode="stack",
                color_discrete_sequence=["#1a76d1", "#e1effa"],
            )
            fig_bar.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Poppins, Inter", color="#2c2d3f"),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="card-header-line">Tier Pool Allocation & Physical Slots</div>
                """,
                unsafe_allow_html=True,
            )
            tier_df = pd.DataFrame([
                {"Tier": k, "Occupied": v["occupied"], "Capacity": v["capacity"]}
                for k, v in sim_results["tier_summary"].items()
            ])
            fig_tier = go.Figure()
            fig_tier.add_trace(go.Bar(
                name="Occupied Slots",
                x=tier_df["Tier"],
                y=tier_df["Occupied"],
                marker_color="#1a76d1",
            ))
            fig_tier.add_trace(go.Bar(
                name="Available Headroom",
                x=tier_df["Tier"],
                y=tier_df["Capacity"] - tier_df["Occupied"],
                marker_color="#f0f7fe",
            ))
            fig_tier.update_layout(
                barmode="stack",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Poppins, Inter", color="#2c2d3f"),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_tier, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Admitted Memory Table
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-header-line">Active In-Memory Telemetry Registry</div>
            """,
            unsafe_allow_html=True,
        )
        living_df = pd.DataFrame(sim_results["living_memory"])

        f1, f2, f3 = st.columns([1, 1, 1.5])
        with f1:
            sel_tier = st.multiselect("Filter Memory Tier", ["EMERGENCY", "HIGH_PRIORITY", "NORMAL"], default=["EMERGENCY", "HIGH_PRIORITY", "NORMAL"])
        with f2:
            sel_care = st.multiselect("Filter Unit Care Level", ["ER", "ICU", "HDU", "WARD"], default=["ER", "ICU", "HDU", "WARD"])
        with f3:
            search_id = st.text_input("Search Patient Identifier", placeholder="Filter by patient ID...")

        filtered_df = living_df[
            living_df["tier"].isin(sel_tier) &
            living_df["care_level"].isin(sel_care)
        ]
        if search_id.strip():
            filtered_df = filtered_df[filtered_df["patient_id"].astype(str).str.contains(search_id.strip(), case=False)]

        display_cols = ["page_id", "patient_id", "tier", "care_level", "criticality", "cms", "severity", "biomarker_risk", "sepsis", "reason"]

        st.dataframe(
            filtered_df[display_cols].sort_values(by="cms", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=300,
        )

        csv_data = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Admitted Telemetry Pages (CSV)",
            data=csv_data,
            file_name="livingmemory_active_pages.csv",
            mime="text/csv",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: REAL-TIME PATIENT TRIAGE
    # ---------------------------------------------------------
    with tab2:
        st.markdown('<div class="anim-fade">', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="margin-bottom:18px;">
                <div style="font-size:1.25rem; font-weight:700; color:#2c2d3f;">Patient Telemetry Admission Evaluation</div>
                <p style="font-size:0.88rem; color:#64748b;">Simulate real-time ingestion of vital signs and biomarkers to assess clinical priority and tier placement.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preset = st.selectbox(
            "Clinical Case Templates:",
            [
                "Custom Clinical Telemetry Input",
                "Profile 1: Sepsis Shock Crisis (ER Escalation)",
                "Profile 2: Post-Surgical High-Risk ICU Patient",
                "Profile 3: Stable Step-Down Ward Patient",
            ],
        )

        preset_vals = {
            "heart_rate_mean": 88.0,
            "spo2_mean": 96.0,
            "respiratory_rate_mean": 18.0,
            "temperature_mean": 37.0,
            "apache_score": 14.0,
            "sofa_score": 4.0,
            "glucose_mean": 115.0,
            "lactate_mean": 2.0,
            "comorbidity_score": 2.0,
            "sepsis_flag": 0,
        }

        if "Sepsis Shock" in preset:
            preset_vals = {
                "heart_rate_mean": 132.0,
                "spo2_mean": 87.0,
                "respiratory_rate_mean": 34.0,
                "temperature_mean": 39.5,
                "apache_score": 36.0,
                "sofa_score": 16.0,
                "glucose_mean": 220.0,
                "lactate_mean": 7.2,
                "comorbidity_score": 4.0,
                "sepsis_flag": 1,
            }
        elif "Post-Surgical" in preset:
            preset_vals = {
                "heart_rate_mean": 108.0,
                "spo2_mean": 92.0,
                "respiratory_rate_mean": 24.0,
                "temperature_mean": 38.3,
                "apache_score": 24.0,
                "sofa_score": 10.0,
                "glucose_mean": 155.0,
                "lactate_mean": 3.9,
                "comorbidity_score": 3.0,
                "sepsis_flag": 0,
            }
        elif "Stable Step-Down" in preset:
            preset_vals = {
                "heart_rate_mean": 72.0,
                "spo2_mean": 99.0,
                "respiratory_rate_mean": 14.0,
                "temperature_mean": 36.7,
                "apache_score": 6.0,
                "sofa_score": 1.0,
                "glucose_mean": 92.0,
                "lactate_mean": 1.1,
                "comorbidity_score": 1.0,
                "sepsis_flag": 0,
            }

        with st.form("triage_form_mediplus"):
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Physiological Vitals</div>", unsafe_allow_html=True)
                hr = st.slider("Heart Rate (bpm)", 40.0, 200.0, float(preset_vals["heart_rate_mean"]), 1.0)
                spo2 = st.slider("SpO2 Saturation (%)", 60.0, 100.0, float(preset_vals["spo2_mean"]), 0.5)
                rr = st.slider("Respiratory Rate (breaths/min)", 8.0, 50.0, float(preset_vals["respiratory_rate_mean"]), 1.0)
                temp = st.slider("Body Temperature (°C)", 32.0, 42.0, float(preset_vals["temperature_mean"]), 0.1)

            with p2:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Acuity & Biomarkers</div>", unsafe_allow_html=True)
                apache = st.slider("APACHE II Score (0 - 71)", 0.0, 60.0, float(preset_vals["apache_score"]), 1.0)
                sofa = st.slider("SOFA Score (0 - 24)", 0.0, 24.0, float(preset_vals["sofa_score"]), 1.0)
                glucose = st.slider("Serum Glucose (mg/dL)", 40.0, 400.0, float(preset_vals["glucose_mean"]), 5.0)
                lactate = st.slider("Serum Lactate (mmol/L)", 0.5, 15.0, float(preset_vals["lactate_mean"]), 0.1)

            with p3:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Patient Metadata</div>", unsafe_allow_html=True)
                pid = st.text_input("Patient Identification", "PT-MEDIPLUS-102")
                comorb = st.slider("Comorbidity Score", 0.0, 10.0, float(preset_vals["comorbidity_score"]), 1.0)
                sepsis = st.radio("Sepsis Clinical Flag", [0, 1], index=int(preset_vals["sepsis_flag"]), format_func=lambda x: "Active Sepsis (1)" if x == 1 else "Non-Septic (0)")

            submit_btn = st.form_submit_button("Evaluate Telemetry & Run Admission Triage", use_container_width=True)

        input_vitals = {
            "heart_rate_mean": hr,
            "spo2_mean": spo2,
            "respiratory_rate_mean": rr,
            "temperature_mean": temp,
            "apache_score": apache,
            "sofa_score": sofa,
            "glucose_mean": glucose,
            "lactate_mean": lactate,
            "comorbidity_score": comorb,
            "sepsis_flag": sepsis,
        }

        triage_res = simulator.triage_single_patient(
            vitals=input_vitals,
            patient_id=pid,
            cms_weights=cms_weights,
            critical_threshold=critical_threshold,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Decision Output Cards in Mediplus Style
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#fe7660;">Mortality Risk</div>
                    <div style="font-size:2rem; font-weight:800; color:#fe7660; margin:6px 0;">{triage_res['criticality'] * 100:.1f}%</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Calibrated Probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#1a76d1;">Care Level</div>
                    <div style="font-size:2rem; font-weight:800; color:#1a76d1; margin:6px 0;">{triage_res['care_level']}</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Bed Escalation Priority</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#0d9488;">CMS Score</div>
                    <div style="font-size:2rem; font-weight:800; color:#0d9488; margin:6px 0;">{triage_res['cms']:.4f}</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Clinical Memory Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r4:
            badge_class = "badge-emergency-coral" if triage_res["tier"] == "EMERGENCY" else ("badge-high-blue" if triage_res["tier"] == "HIGH_PRIORITY" else "badge-normal-teal")
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#2c2d3f;">Assigned Tier</div>
                    <div style="margin:10px 0;"><span class="{badge_class}">{triage_res['tier']}</span></div>
                    <div style="font-size:0.75rem; color:#8898aa;">Replacement Pool</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="background:#f4f9fd; border:1px solid #d0e7f9; border-left:4px solid #1a76d1; border-radius:8px; padding:16px 20px; margin-top:16px;">
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#1a76d1; margin-bottom:2px;">Clinical Preservation Rationale</div>
                <div style="font-size:0.95rem; font-weight:600; color:#2c2d3f;">{triage_res['reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: MODEL VALIDATION
    # ---------------------------------------------------------
    with tab3:
        st.markdown('<div class="anim-fade">', unsafe_allow_html=True)
        metrics = simulator.classifier.metrics

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#1a76d1;">Accuracy Score</div>
                    <div style="font-size:1.9rem; font-weight:800; color:#1a76d1; margin:4px 0;">{metrics['accuracy'] * 100:.2f}%</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Stratified Holdout Split</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#0d9488;">ROC - AUC Score</div>
                    <div style="font-size:1.9rem; font-weight:800; color:#0d9488; margin:4px 0;">{metrics['roc_auc']:.4f}</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Discrimination Metric</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#2c2d3f;">PR - AUC Score</div>
                    <div style="font-size:1.9rem; font-weight:800; color:#2c2d3f; margin:4px 0;">{metrics['pr_auc']:.4f}</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Average Precision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="card-header-line">Physiological Feature Importances</div>
                """,
                unsafe_allow_html=True,
            )
            feat_df = pd.DataFrame(
                list(metrics["feature_importances"].items()),
                columns=["Physiological Feature", "Gini Importance"],
            ).sort_values(by="Gini Importance", ascending=True)

            fig_feat = px.bar(
                feat_df,
                x="Gini Importance",
                y="Physiological Feature",
                orientation="h",
                color="Gini Importance",
                color_continuous_scale=["#e8f3fc", "#1a76d1", "#135da7"],
            )
            fig_feat.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Poppins, Inter", color="#2c2d3f"),
                height=320,
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="card-header-line">Confusion Matrix</div>
                """,
                unsafe_allow_html=True,
            )
            cm = metrics["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Mortality", y="Actual Mortality", color="Count"),
                x=["Survives (0)", "Deceased (1)"],
                y=["Survives (0)", "Deceased (1)"],
                text_auto=True,
                color_continuous_scale=["#ffffff", "#e8f3fc", "#1a76d1"],
            )
            fig_cm.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Poppins, Inter", color="#2c2d3f"),
                height=320,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-header-line">Detailed Classification Report</div>
            """,
            unsafe_allow_html=True,
        )
        rep_df = pd.DataFrame(metrics["report"]).transpose()
        st.dataframe(rep_df.style.format(precision=3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
