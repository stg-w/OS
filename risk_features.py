import pandas as pd

# Load master dataset with explicit string types for careunit columns to fix the warning
master = pd.read_csv(
    "master.csv", 
    dtype={"first_careunit": str, "last_careunit": str}
)

master["age_risk"] = (
    master["anchor_age"] / 100
).clip(0, 1)

master["los"] = master["los"].fillna(0)

master["icu_risk"] = (
    master["los"] / 10
).clip(0, 1)

y = master["hospital_expire_flag"]

# Load transfers with explicit data types for matching
transfers = pd.read_csv(
    "transfers_sample.csv",
    dtype={"subject_id": int}
)

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

master["escalation_risk"] = (
    master["transfer_count"] / 10
).clip(0, 1)

# Load lab events with explicit data types for matching and flag checking
labs = pd.read_csv(
    "labevents_sample.csv",
    dtype={"subject_id": int, "flag": str}
)

labs["abnormal"] = (
    labs["flag"]
    .notna()
    .astype(int)
)

abnormal_counts = (
    labs
    .groupby("subject_id")
    ["abnormal"]
    .sum()
    .reset_index()
)

master = master.merge(
    abnormal_counts,
    on="subject_id",
    how="left"
)

master["abnormal"] = (
    master["abnormal"]
    .fillna(0)
)

master["biomarker_risk"] = (
    master["abnormal"] / 20
).clip(0, 1)

master.to_csv(
    "features.csv",
    index=False
)

print("features.csv created")
print(master.columns.tolist())
