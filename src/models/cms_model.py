"""
Clinical Memory Score (CMS) Generation Module

Calculates the Clinical Memory Score (CMS) for each patient by combining
multiple risk indicators, including AI-predicted mortality risk, biomarker
abnormalities, escalation history, ICU severity, age-related risk, and
inheritance priority. Each factor contributes according to predefined
weights to produce a single priority score representing the clinical
importance of retaining a patient's data in memory. The generated CMS
values are stored and exported for use in LivingMemoryOS memory allocation,
patient prioritization, and replacement decision simulations.
"""

import pandas as pd

master = pd.read_csv(
    r"C:\Users\subiw\OS\data\inheritance_output.csv"
)

# ==========================================
# CMS
# ==========================================

master["cms"] = (

    0.30 * master["ai_risk"]

    +

    0.20 * master["biomarker_risk"]

    +

    0.20 * master["escalation_risk"]

    +

    0.15 * master["icu_risk"]

    +

    0.10 * master["age_risk"]

    +

    0.05 * master["inheritance"]

)

master.to_csv(
    "cms_output.csv",
    index=False
)

print("Saved: cms_output.csv")