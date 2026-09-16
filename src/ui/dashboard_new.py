import streamlit as st
import pandas as pd
import plotly.express as px


def render_dashboard():

    st.set_page_config(
        page_title="LivingMemoryOS",
        page_icon="🧠",
        layout="wide"
    )

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

    results = pd.read_csv("livingmemory_results_v4.csv")

    # ==========================================
    # SIDEBAR
    # ==========================================

    st.sidebar.title("⚙ System Controls")

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

    if "care_level" in results.columns:

        care_units = st.sidebar.multiselect(
            "Care Units",
            sorted(results["care_level"].dropna().unique()),
            default=list(
                results["care_level"].dropna().unique()
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

    filtered = filtered[
        filtered["cms"] >= cms_threshold
    ]

    if len(care_units) > 0:
        filtered = filtered[
            filtered["care_level"].isin(care_units)
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
    )

    avg_cms = round(
        filtered["cms"].mean(),
        3
    )

    max_cms = round(
        filtered["cms"].max(),
        3
    )

    c1,c2,c3,c4 = st.columns(4)

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

    left,right = st.columns(2)

    with left:

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

        if "care_level" in filtered.columns:

            st.subheader(
                "Care Unit Distribution"
            )

            care_counts = (
                filtered["care_level"]
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

    if "ai_risk" in filtered.columns:

        st.subheader(
            "AI Mortality Risk Distribution"
        )

        fig3 = px.scatter(
            filtered,
            x="ai_risk",
            y="cms",
            color="cms",
            size="cms",
            hover_data=filtered.columns
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
            "cms",
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

        "Method":[
            "FIFO",
            "LivingMemoryOS"
        ],

        "Critical Patients Retained":[
            min(10,total_patients),
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
            "cms",
            ascending=False
        )
        .head(5)
    )

    for _,row in cards.iterrows():

        st.markdown(f"""
        <div class="patient-card">

        <h4>Priority Patient</h4>

        CMS: {row['cms']:.3f}<br>

        AI Risk: {row.get('ai_risk',0):.3f}<br>

        Biomarker Risk:
        {row.get('biomarker_risk',0):.3f}<br>

        Escalation Risk:
        {row.get('escalation_risk',0):.3f}<br>

        ICU Risk:
        {row.get('icu_risk',0):.3f}

        </div>
        """,
        unsafe_allow_html=True)

    # ==========================================
    # DOWNLOAD
    # ==========================================

    st.download_button(
        "⬇ Download Results CSV",
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