import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# Load master dataset with explicit string types for careunit columns to fix the warning
master_path = os.path.join(DATA_DIR, "master.csv") if os.path.exists(os.path.join(DATA_DIR, "master.csv")) else "master.csv"
master = pd.read_csv(
    r"C:\Users\subiw\OS\data\master.csv", 
    low_memory=False
    master_path, 
    low_memory=False,
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
transfers_path = os.path.join(DATA_DIR, "transfers_sample.csv") if os.path.exists(os.path.join(DATA_DIR, "transfers_sample.csv")) else "transfers_sample.csv"
transfers = pd.read_csv(
    r"C:\Users\subiw\OS\data\transfers_sample.csv",
    low_memory=False
    transfers_path,
    low_memory=False,
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
labs_path = os.path.join(DATA_DIR, "labevents_sample.csv") if os.path.exists(os.path.join(DATA_DIR, "labevents_sample.csv")) else "labevents_sample.csv"
labs = pd.read_csv(
    r"C:\Users\subiw\OS\data\labevents_sample.csv",
    low_memory=False
    labs_path,
    low_memory=False,
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

out_features = os.path.join(DATA_DIR, "features.csv")
master.to_csv(
    "features.csv",
    out_features,
    index=False
)

print("features.csv created")
print(f"features.csv created at {out_features}")
print(master.columns.tolist())
