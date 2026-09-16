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