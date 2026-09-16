import pandas as pd

# =====================================
# LOAD DATA
# =====================================

patients = pd.read_csv(
    "patients_sample.csv"
)

admissions = pd.read_csv(
    "admissions_sample.csv"
)

icu = pd.read_csv(
    "icustays_sample.csv"
)

transfers = pd.read_csv(
    "transfers_sample.csv"
)

# =====================================
# BASE MERGE
# =====================================

master = admissions.merge(
    patients,
    on="subject_id",
    how="left"
)

master = master.merge(
    icu[
        [
            "subject_id",
            "hadm_id",
            "los",
            "first_careunit",
            "last_careunit"
        ]
    ],
    on=["subject_id", "hadm_id"],
    how="left"
)

# =====================================
# TRANSFER FEATURES
# =====================================

transfer_counts = (
    transfers
    .groupby("subject_id")
    .size()
    .reset_index(name="transfer_count")
)

master = master.merge(
    transfer_counts,
    on="subject_id",
    how="left"
)

master["transfer_count"] = (
    master["transfer_count"]
    .fillna(0)
)

# =====================================
# AGE RISK
# =====================================

master["age_risk"] = (
    master["anchor_age"] / 100
).clip(0, 1)

# =====================================
# ICU RISK
# =====================================

master["los"] = (
    master["los"]
    .fillna(0)
)

master["icu_risk"] = (
    master["los"] / 10
).clip(0, 1)

# =====================================
# ESCALATION RISK
# =====================================

master["escalation_risk"] = (
    master["transfer_count"] / 10
).clip(0, 1)

# =====================================
# SAVE
# =====================================

master.to_csv(
    "master.csv",
    index=False
)

# =====================================
# SUMMARY
# =====================================

print("\nmaster.csv created successfully")

print(
    "\nRows:",
    len(master)
)

print(
    "Columns:",
    len(master.columns)
)

print(
    "\nColumns:"
)

for col in master.columns:
    print(col)

print(
    "\nAverage Age Risk:",
    round(
        master["age_risk"].mean(),
        3
    )
)

print(
    "Average ICU Risk:",
    round(
        master["icu_risk"].mean(),
        3
    )
)

print(
    "Average Escalation Risk:",
    round(
        master["escalation_risk"].mean(),
        3
    )
)

master = master.drop(
    columns=["abnormal"],
    errors="ignore"
)