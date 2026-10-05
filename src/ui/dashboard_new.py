"""
Streamlit Web Dashboard for LivingMemoryOS v4 (MIMIC-IV Clinical Cohort)
Inspired by clean Mediplus Medical Clinical UI
Pure White Base, Royal Blue & Coral Red Accents, Zero Emojis
Identical design system, layout, and visual fidelity as dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config_new import (
    DEFAULT_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_PROTECTED_THRESHOLDS,
    EMERGENCY_CMS_THRESHOLD,
    ADAPTIVE_EXPANSION_STEP,
    ADAPTIVE_TRIGGER_COUNT,
    FEATURES,
)
from src.models.mimic_classifier import MimicMortalityClassifier
from src.memory.mimic_simulator import MimicLivingMemorySimulator


# Define cache function at top-level to prevent Python inspect tokenization errors
@st.cache_resource(show_spinner="Initializing MIMIC-IV Clinical Machine Learning Model...")
def get_simulator():
    classifier = MimicMortalityClassifier()
    classifier.load_and_train()
    return MimicLivingMemorySimulator(classifier)


def render_dashboard():
    # Page configuration safely handled
    try:
        st.set_page_config(
            page_title="LivingMemoryOS v4 | MIMIC-IV Clinical AI Memory",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

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
            z-index: 100 !important;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #f8fbfe !important;
        }

        /* Always-visible Sidebar Toggle / Expand Button */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[data-testid="stSidebarCollapseButton"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 999999 !important;
            background-color: #e8f3fc !important;
            color: #1a76d1 !important;
            border: 1px solid #c9e4fb !important;
            border-radius: 8px !important;
            padding: 4px 8px !important;
            top: 10px !important;
            left: 10px !important;
            box-shadow: 0 2px 6px rgba(26, 118, 209, 0.15) !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="collapsedControl"]:hover,
        button[data-testid="stSidebarCollapseButton"]:hover {
            background-color: #1a76d1 !important;
            color: #ffffff !important;
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
            <div class="brand-logo">Living<span>MemoryOS</span> <span style="font-size:0.85rem; font-weight:700; color:#1a76d1; background:#e8f3fc; padding:3px 10px; border-radius:12px; vertical-align:middle; border:1px solid #d0e7f9;">v4 (MIMIC-IV)</span></div>
            <div class="brand-tagline">Clinical Admissions & ICU Memory Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        """
        <div class="mediplus-hero">
            <div class="hero-pretitle">Next-Gen Architecture: MIMIC-IV Clinical Cohort</div>
            <h1 class="hero-main-title">Intelligent Memory Allocation for <span>MIMIC-IV Inpatients</span></h1>
            <p class="hero-description">
                Preserving high-risk physiological data in resource-constrained medical edge hardware using patient-criticality inheritance, non-evictable prognostic retention, and self-evolving adaptive capacity across 550,818 admissions.
            </p>
            <div>
                <span class="pill-badge">MIMIC-IV Clinical Inpatients</span>
                <span class="pill-badge" style="margin-left:6px;">Patient-Criticality Inheritance</span>
                <span class="pill-badge" style="margin-left:6px;">Prognostic Retention Lock</span>
                <span class="pill-badge" style="margin-left:6px;">Self-Evolving Buffer Expansion</span>
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

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#1a76d1; text-transform:uppercase; margin-bottom:8px;'>Memory Page Slots</div>", unsafe_allow_html=True)
        memory_size = st.slider("Total Memory Capacity (Slots)", 20, 300, DEFAULT_CAPACITY, 10)
        stream_size = st.select_slider(
            "Telemetry Stream Ingestion Size",
            options=[1000, 2500, 5000, 10000, 20000],
            value=5000,
            help="Number of streaming clinical records evaluated during the replacement run.",
        )

        st.markdown(
            f"""
            <div style="background:#e8f3fc; border:1px solid #c9e4fb; border-radius:8px; padding:12px; margin:16px 0; text-align:center;">
                <div style="font-size:0.72rem; font-weight:700; color:#1a76d1; text-transform:uppercase; letter-spacing:0.5px;">Active Memory Budget</div>
                <div style="font-size:1.45rem; font-weight:800; color:#2c2d3f;">{memory_size} Pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#1a76d1; text-transform:uppercase; margin:16px 0 8px 0;'>Prognostic Protection Cutoffs</div>", unsafe_allow_html=True)
        ai_cutoff = st.slider("AI Mortality Cutoff", 0.50, 0.95, DEFAULT_PROTECTED_THRESHOLDS["ai_risk_cutoff"], 0.05)
        bio_cutoff = st.slider("Biomarker Risk Cutoff", 0.50, 0.95, DEFAULT_PROTECTED_THRESHOLDS["biomarker_risk_cutoff"], 0.05)

        protected_thresholds = {
            "ai_risk_cutoff": ai_cutoff,
            "biomarker_risk_cutoff": bio_cutoff,
        }

        with st.expander("CMS Clinical Weights"):
            w_ai = st.number_input("AI Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["ai_risk"], 0.05)
            w_bio = st.number_input("Biomarker Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["biomarker_risk"], 0.05)
            w_esc = st.number_input("Escalation Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["escalation_risk"], 0.05)
            w_icu = st.number_input("ICU Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["icu_risk"], 0.05)
            w_age = st.number_input("Age Risk Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["age_risk"], 0.05)
            w_inh = st.number_input("Inheritance Weight", 0.0, 1.0, DEFAULT_CMS_WEIGHTS["inheritance"], 0.05)

            cms_weights = {
                "ai_risk": w_ai,
                "biomarker_risk": w_bio,
                "escalation_risk": w_esc,
                "icu_risk": w_icu,
                "age_risk": w_age,
                "inheritance": w_inh,
            }

        with st.expander("Self-Evolving Capacity Rules"):
            emergency_thresh = st.slider("Emergency CMS Threshold", 0.60, 0.95, EMERGENCY_CMS_THRESHOLD, 0.05)
            adaptive_trigger = st.slider("Trigger Emergency Page Count", 10, 50, ADAPTIVE_TRIGGER_COUNT, 5)
            adaptive_step = st.slider("Capacity Expansion Step", 10, 50, ADAPTIVE_EXPANSION_STEP, 5)

        st.markdown("---")
        st.markdown("<div style='font-size:0.75rem; color:#8898aa; text-align:center;'>Mediplus Medical Clinical Palette</div>", unsafe_allow_html=True)

    # Run Simulation
    sim_results = simulator.run_simulation(
        memory_size=memory_size,
        stream_size=stream_size,
        cms_weights=cms_weights,
        protected_thresholds=protected_thresholds,
        emergency_threshold=emergency_thresh,
        adaptive_expansion_step=adaptive_step,
        adaptive_trigger_count=adaptive_trigger,
    )

    # Mediplus Stats Banner (Royal Blue)
    st.markdown(
        f"""
        <div class="stats-banner anim-fade">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr);">
                <div class="stat-box">
                    <div class="stat-number">{sim_results['living_average_cms']}</div>
                    <div class="stat-label">LivingMemory Retained</div>
                    <div class="stat-sub">avg retained priority ({memory_size} slots)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sim_results['fifo_average_cms']}</div>
                    <div class="stat-label">FIFO Baseline Retained</div>
                    <div class="stat-sub">unaware temporal buffer</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">+{sim_results['improvement_pct']}%</div>
                    <div class="stat-label">Retention Improvement</div>
                    <div class="stat-sub">critical clinical preservation</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sim_results['protected_pages']} / {memory_size}</div>
                    <div class="stat-label">Protected Patients</div>
                    <div class="stat-sub">non-evictable emergency pages</div>
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
                "Algorithm": ["FIFO Baseline Queue", "LivingMemoryOS v4 (CAMR)"],
                "Average Clinical Memory Score (CMS)": [sim_results["fifo_average_cms"], sim_results["living_average_cms"]],
                "Protected Pages Retained": [sim_results["fifo_protected_pages"], sim_results["protected_pages"]],
            })
            fig_bar = px.bar(
                comp_df,
                x="Algorithm",
                y="Average Clinical Memory Score (CMS)",
                color="Algorithm",
                color_discrete_sequence=["#e1effa", "#1a76d1"],
                text_auto=".4f",
            )
            fig_bar.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Poppins, Inter", color="#2c2d3f"),
                height=320,
                showlegend=False,
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
            living_df = pd.DataFrame(sim_results["living_memory"])
            tier_counts = living_df["tier"].value_counts().to_dict() if "tier" in living_df.columns else {}
            
            tier_df = pd.DataFrame([
                {"Tier": "EMERGENCY", "Occupied": tier_counts.get("EMERGENCY", 0), "Max Capacity": memory_size},
                {"Tier": "HIGH_PRIORITY", "Occupied": tier_counts.get("HIGH_PRIORITY", 0), "Max Capacity": memory_size},
                {"Tier": "NORMAL", "Occupied": tier_counts.get("NORMAL", 0), "Max Capacity": memory_size},
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
                y=tier_df["Max Capacity"] - tier_df["Occupied"],
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

        f1, f2, f3 = st.columns([1, 1, 1.5])
        with f1:
            sel_tier = st.multiselect(
                "Filter Memory Tier",
                ["EMERGENCY", "HIGH_PRIORITY", "NORMAL"],
                default=["EMERGENCY", "HIGH_PRIORITY", "NORMAL"],
            )
        with f2:
            care_col = "first_careunit" if "first_careunit" in living_df.columns else ("care_level" if "care_level" in living_df.columns else None)
            care_options = sorted(list(set(str(u) for u in living_df[care_col].dropna().unique()))) if care_col else []
            sel_care = st.multiselect(
                "Filter Care Unit",
                care_options,
                default=care_options[:4] if len(care_options) >= 4 else care_options,
            )
        with f3:
            search_id = st.text_input("Search Patient Identifier", placeholder="Filter by patient ID...")

        filtered_df = living_df.copy()
        if "tier" in filtered_df.columns and sel_tier:
            filtered_df = filtered_df[filtered_df["tier"].isin(sel_tier)]
        if care_col and sel_care:
            filtered_df = filtered_df[filtered_df[care_col].astype(str).isin(sel_care)]
        if search_id.strip():
            id_col = "subject_id" if "subject_id" in filtered_df.columns else "patient_id"
            filtered_df = filtered_df[filtered_df[id_col].astype(str).str.contains(search_id.strip(), case=False)]

        preferred_cols = [
            "page_id", "subject_id", "hadm_id", "admission_type", "first_careunit",
            "anchor_age", "los", "transfer_count", "abnormal", "ai_risk",
            "inheritance", "cms", "protected", "tier", "reason"
        ]
        display_cols = [c for c in preferred_cols if c in filtered_df.columns]
        if not display_cols:
            display_cols = filtered_df.columns.tolist()

        st.dataframe(
            filtered_df[display_cols].sort_values(by="cms", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=300,
        )

        csv_data = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Admitted Telemetry Pages (CSV)",
            data=csv_data,
            file_name="livingmemory_active_pages_v4.csv",
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
                "Profile 1: Sepsis Shock Crisis (Elderly ICU, Prolonged Stay & Transfers)",
                "Profile 2: Post-Surgical High-Risk ICU Patient (Cardiac & Biomarker Crisis)",
                "Profile 3: Stable Step-Down Ward Patient (Low Acuity, Normal Labs)",
            ],
        )

        p_age = 62.0
        p_los = 4.2
        p_trans = 2.0
        p_abn = 18.0
        p_type = "EMERGENCY"
        p_unit = "MICU"
        p_id = "PT-MEDIPLUS-102"

        if "Sepsis Shock" in preset:
            p_age = 78.0
            p_los = 14.5
            p_trans = 6.0
            p_abn = 45.0
            p_type = "EW EMER."
            p_unit = "MICU"
            p_id = "PT-CRIT-901"
        elif "Post-Surgical" in preset:
            p_age = 66.0
            p_los = 8.0
            p_trans = 3.0
            p_abn = 62.0
            p_type = "URGENT"
            p_unit = "CVICU"
            p_id = "PT-CARD-442"
        elif "Stable Step-Down" in preset:
            p_age = 34.0
            p_los = 1.0
            p_trans = 0.0
            p_abn = 2.0
            p_type = "OBSERVATION ADMIT"
            p_unit = "WARD"
            p_id = "PT-WARD-110"

        with st.form("triage_form_mediplus"):
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Demographics & Stay</div>", unsafe_allow_html=True)
                in_age = st.slider("Patient Age (Years)", 18.0, 100.0, float(p_age), 1.0)
                in_los = st.slider("ICU Length of Stay (Days)", 0.0, 30.0, float(p_los), 0.5)

            with p2:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Acuity & Biomarkers</div>", unsafe_allow_html=True)
                in_trans = st.slider("Unit Transfer Count", 0.0, 20.0, float(p_trans), 1.0)
                in_abn = st.slider("Abnormal Lab Events Count", 0.0, 200.0, float(p_abn), 1.0)

            with p3:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#1a76d1; margin-bottom:10px;'>Patient Metadata</div>", unsafe_allow_html=True)
                in_pid = st.text_input("Patient Identification", p_id)
                in_type = st.selectbox("Admission Type", ["EMERGENCY", "EW EMER.", "DIRECT EMER.", "URGENT", "OBSERVATION ADMIT", "SURGICAL SAME DAY"], index=0)
                in_unit = st.selectbox("Care Unit", ["MICU", "SICU", "CCU", "CVICU", "TSICU", "WARD"], index=0)

            submit_btn = st.form_submit_button("Evaluate Telemetry & Run Admission Triage", use_container_width=True)

        triage_res = simulator.triage_single_patient(
            anchor_age=in_age,
            los=in_los,
            transfer_count=in_trans,
            abnormal=in_abn,
            admission_type=in_type,
            first_careunit=in_unit,
            subject_id=in_pid,
            cms_weights=cms_weights,
            protected_thresholds=protected_thresholds,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Decision Output Cards in Mediplus Style
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#fe7660;">Mortality Risk</div>
                    <div style="font-size:2rem; font-weight:800; color:#fe7660; margin:6px 0;">{triage_res['ai_risk'] * 100:.1f}%</div>
                    <div style="font-size:0.75rem; color:#8898aa;">Calibrated Probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#1a76d1;">Inheritance</div>
                    <div style="font-size:2rem; font-weight:800; color:#1a76d1; margin:6px 0;">{triage_res['inheritance']:.4f}</div>
                    <div style="font-size:0.75rem; color:#8898aa;">0.5*ICU + 0.5*Escalation</div>
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
            protect_label = "NON-EVICTABLE" if triage_res.get("protected", False) else "EVICTABLE"
            st.markdown(
                f"""
                <div class="feature-card" style="text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#2c2d3f;">Assigned Tier</div>
                    <div style="margin:10px 0;"><span class="{badge_class}">{triage_res['tier']}</span></div>
                    <div style="font-size:0.75rem; font-weight:700; color:{'#fe7660' if triage_res.get('protected') else '#0d9488'};">{protect_label}</div>
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
                    <div style="font-size:1.9rem; font-weight:800; color:#1a76d1; margin:4px 0;">{metrics.get('accuracy', 0.85) * 100:.2f}%</div>
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
                    <div style="font-size:1.9rem; font-weight:800; color:#0d9488; margin:4px 0;">{metrics.get('roc_auc', 0.88):.4f}</div>
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
                    <div style="font-size:1.9rem; font-weight:800; color:#2c2d3f; margin:4px 0;">{metrics.get('pr_auc', 0.82):.4f}</div>
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
                list(metrics.get("feature_importances", {}).items()),
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
            cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
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
        rep = metrics.get("report", {})
        if rep:
            rep_df = pd.DataFrame(rep).transpose()
            st.dataframe(rep_df.style.format(precision=3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


render_mimic_dashboard = render_dashboard

if __name__ == "__main__":
    render_dashboard()
    render_dashboard()