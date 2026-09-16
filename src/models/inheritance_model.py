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