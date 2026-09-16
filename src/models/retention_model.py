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