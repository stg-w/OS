"""
Prognostic Retention Classification Module

This module applies LivingMemoryOS prognostic retention policies to identify
patients that require protected memory retention and elevated clinical
attention. Using previously computed AI Risk and Clinical Memory Score (CMS)
values, it classifies patients as protected when mortality risk or biomarker
severity exceeds predefined thresholds and marks high-priority patients based
on overall clinical significance. The generated retention labels are used by
the memory management system to prevent critical patient records from being
evicted and to ensure that clinically important cases remain accessible during
resource-constrained memory operations.
"""

import pandas as pd

master = pd.read_csv(
    r"C:\Users\subiw\OS\data\cms_output.csv",
    low_memory=False
)

# ======================================
# PROGNOSTIC RETENTION
# ======================================

master["protected"] = (
    (master["ai_risk"] >= 0.80)
    |
    (master["biomarker_risk"] >= 0.80)
)

master["priority"] = (
    master["cms"] >= 0.70
)

master.to_csv(
    r"C:\Users\subiw\OS\data\retention_output.csv",
    index=False
)

print("\nSaved: retention_output.csv")

print(
    "\nProtected Patients:",
    master["protected"].sum()
)

print(
    "Priority Patients:",
    master["priority"].sum()
)