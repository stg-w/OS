"""
Patient Inheritance Score Generation Module

Calculates an inheritance score for each patient by combining ICU risk and
escalation risk into a single continuity-of-care metric. The inheritance
score represents the likelihood that a patient's clinical importance should
persist over time based on the severity of their ICU stay and the frequency
of care transitions. This derived feature is later incorporated into the
Clinical Memory Score (CMS) to help LivingMemoryOS retain patient records
that demonstrate sustained clinical significance.
"""

import pandas as pd

master = pd.read_csv(
    r"C:\Users\subiw\OS\data\model_output.csv"
)

# ==========================================
# PATIENT CRITICALITY INHERITANCE
# ==========================================

master["inheritance"] = (
    0.5 * master["icu_risk"]
    +
    0.5 * master["escalation_risk"]
)

master.to_csv(
    "inheritance_output.csv",
    index=False
)

print("Saved: inheritance_output.csv")