"""
Streamlit Web Dashboard Module for LivingMemoryOS v4 (MIMIC-IV Clinical Cohort)
Clean Medical Clinical UI: Pure White Base, Royal Blue & Coral Red Accents.
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.ui.mimic_dashboard import render_mimic_dashboard


def render_dashboard():
    """Render the LivingMemoryOS v4 MIMIC-IV Clinical Dashboard."""
    render_mimic_dashboard()
    try:
        st.set_page_config(
            page_title="LivingMemoryOS",
            page_icon="🧠",
            layout="wide"
        )
    except Exception:
        pass

    # ==========================================
    # STYLING
    # ==========================================

    st.markdown("""
    <style>

    .stApp{
        background-color:#ffe4ec;
    }

    section[data-testid="stSidebar"]{
        background-color:#ffd6e7;
    }

    h1,h2,h3{
        color:#6d214f;
    }

    div[data-testid="metric-container"]{
        background:white;
        padding:15px;
        border-radius:15px;
        box-shadow:0px 3px 10px rgba(0,0,0,0.08);
    }

    .hero{
        background:white;
        padding:35px;
        border-radius:25px;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom:20px;
    }

    .patient-card{
        background:white;
        padding:15px;
        border-radius:15px;
        margin-bottom:10px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # LOAD DATA
    # ==========================================

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    candidates = [
        os.path.join(data_dir, "livingmemory_results_new.csv"),
        os.path.join("data", "livingmemory_results_new.csv"),
        os.path.join(data_dir, "livingmemory_results_v4.csv"),
        os.path.join("data", "livingmemory_results_v4.csv"),
        "livingmemory_results_new.csv",
        "livingmemory_results_v4.csv",
    ]
    results_path = None
    for c in candidates:
        if os.path.exists(c):
            results_path = c
            break

    if results_path is None:
        st.error("No livingmemory_results file found in data/ or root directory.")
        return

    results = pd.read_csv(results_path)

    # ==========================================
    # SIDEBAR
    # ==========================================

    st.sidebar.title("⚙️ System Controls")

    cms_threshold = st.sidebar.slider(
        "CMS Threshold",
        0.0,
        1.0,
        0.50,
        0.05
    )

    memory_budget = st.sidebar.slider(
        "Memory Slots",
        50,
        500,
        100,
        10
    )

    care_units = []
    unit_col = "first_careunit" if "first_careunit" in results.columns else ("care_level" if "care_level" in results.columns else None)

    if unit_col and unit_col in results.columns:
        care_units = st.sidebar.multiselect(
            "Care Units",
            sorted(results[unit_col].dropna().unique()),
            default=list(
                results[unit_col].dropna().unique()
            )
        )

    st.sidebar.metric(
        "Memory Capacity",
        memory_budget
    )

    # ==========================================
    # FILTERS
    # ==========================================

    filtered = results.copy()

    if "cms" in filtered.columns:
        filtered = filtered[
            filtered["cms"] >= cms_threshold
        ]

    if unit_col and len(care_units) > 0:
        filtered = filtered[
            filtered[unit_col].isin(care_units)
        ]

    # ==========================================
    # HERO SECTION
    # ==========================================

    st.markdown("""
    <div class="hero">

    <h1>🧠 LivingMemoryOS</h1>

    <h3>
    Self-Evolving Clinical Memory Architecture
    </h3>

    <p>
    Prognostic Retention • Criticality Inheritance •
    AI Mortality Prediction • Dynamic Memory Evolution
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # KPI ROW
    # ==========================================

    total_patients = len(filtered)

    critical_patients = len(
        filtered[
            filtered["cms"] > 0.70
        ]
    ) if "cms" in filtered.columns else 0

    avg_cms = round(
        filtered["cms"].mean(),
        3
    ) if "cms" in filtered.columns else 0.0

    max_cms = round(
        filtered["cms"].max(),
        3
    ) if "cms" in filtered.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Patients",
        total_patients
    )

    c2.metric(
        "Critical Patients",
        critical_patients
    )

    c3.metric(
        "Average CMS",
        avg_cms
    )

    c4.metric(
        "Highest CMS",
        max_cms
    )

    st.divider()

    # ==========================================
    # ANALYTICS ROW
    # ==========================================

    left, right = st.columns(2)

    with left:
        if "cms" in filtered.columns:
            st.subheader(
                "Clinical Memory Score Distribution"
            )

            fig = px.histogram(
                filtered,
                x="cms",
                nbins=25,
                color_discrete_sequence=["#ff66a3"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:
        if unit_col and unit_col in filtered.columns:
            st.subheader(
                "Care Unit Distribution"
            )

            care_counts = (
                filtered[unit_col]
                .value_counts()
                .reset_index()
            )

            care_counts.columns = [
                "Care Unit",
                "Count"
            ]

            fig2 = px.bar(
                care_counts,
                x="Care Unit",
                y="Count",
                color="Care Unit"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # ==========================================
    # AI RISK ANALYSIS
    # ==========================================

    if "ai_risk" in filtered.columns and "cms" in filtered.columns:
        st.subheader(
            "AI Mortality Risk Distribution"
        )

        fig3 = px.scatter(
            filtered,
            x="ai_risk",
            y="cms",
            color="cms",
            size="cms" if (filtered["cms"] > 0).any() else None,
            hover_data=[c for c in ["subject_id", "hadm_id", "cms", "ai_risk", "biomarker_risk", "icu_risk"] if c in filtered.columns]
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ==========================================
    # TOP PATIENTS
    # ==========================================

    st.subheader(
        "🚨 Highest Priority Patients"
    )

    top_patients = (
        filtered
        .sort_values(
            "cms" if "cms" in filtered.columns else filtered.columns[0],
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top_patients,
        use_container_width=True,
        height=450
    )

    # ==========================================
    # MEMORY RETENTION
    # ==========================================

    st.subheader(
        "Memory Retention Comparison"
    )

    comparison = pd.DataFrame({
        "Method": [
            "FIFO",
            "LivingMemoryOS"
        ],
        "Critical Patients Retained": [
            min(10, total_patients),
            critical_patients
        ]
    })

    fig4 = px.bar(
        comparison,
        x="Method",
        y="Critical Patients Retained",
        color="Method"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ==========================================
    # TOP MEMORY PAGES
    # ==========================================

    st.subheader(
        "⭐ Highest Priority Memory Pages"
    )

    cards = (
        filtered
        .sort_values(
            "cms" if "cms" in filtered.columns else filtered.columns[0],
            ascending=False
        )
        .head(5)
    )

    for _, row in cards.iterrows():
        cms_val = row.get('cms', 0)
        ai_val = row.get('ai_risk', 0)
        bio_val = row.get('biomarker_risk', 0)
        esc_val = row.get('escalation_risk', 0)
        icu_val = row.get('icu_risk', 0)

        st.markdown(f"""
        <div class="patient-card">
        <h4>Priority Patient {row.get('subject_id', '')}</h4>
        CMS: {cms_val:.3f}<br>
        AI Risk: {ai_val:.3f}<br>
        Biomarker Risk: {bio_val:.3f}<br>
        Escalation Risk: {esc_val:.3f}<br>
        ICU Risk: {icu_val:.3f}
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # DOWNLOAD
    # ==========================================

    st.download_button(
        "📥 Download Results CSV",
        filtered.to_csv(index=False),
        file_name="livingmemory_export.csv",
        mime="text/csv"
    )

    # ==========================================
    # RAW DATA
    # ==========================================

    with st.expander(
        "View Complete Dataset"
    ):
        st.dataframe(
            filtered,
            use_container_width=True
        )


render_mimic_dashboard = render_dashboard

if __name__ == "__main__":
    render_dashboard()
