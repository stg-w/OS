import pandas as pd

# =====================================
# LOAD MASTER
# =====================================

master = pd.read_csv(
    r"C:\Users\subiw\OS\data\master.csv",
    dtype={
        "first_careunit": str,
        "last_careunit": str
    },
    low_memory=False
)

# =====================================
# LOAD LABS
# =====================================

labs = pd.read_csv(
    r"C:\Users\subiw\OS\data\labevents_sample.csv",
    low_memory=False
)

print("Labs loaded:", len(labs))

# =====================================
# CREATE ABNORMAL FLAG
# =====================================

labs["abnormal_lab"] = (
    labs["flag"]
    .notna()
    .astype(int)
)

# =====================================
# COUNT ABNORMAL LABS
# =====================================

abnormal_counts = (
    labs.groupby("subject_id")["abnormal_lab"]
    .sum()
    .reset_index()
)

print(abnormal_counts.head())

# =====================================
# MERGE
# =====================================

master = master.merge(
    abnormal_counts,
    on="subject_id",
    how="left"
)

# =====================================
# FILL MISSING
# =====================================

master["abnormal_lab"] = (
    master["abnormal_lab"]
    .fillna(0)
)

# =====================================
# BIOMARKER RISK
# =====================================

master["biomarker_risk"] = (
    master["abnormal_lab"] / 20
).clip(0, 1)

# =====================================
# SAVE
# =====================================

master.to_csv(
    r"C:\Users\subiw\OS\data\master.csv",
    index=False
)

print("\nbiomarker_risk added")
print(
    master[
        [
            "subject_id",
            "abnormal_lab",
            "biomarker_risk"
        ]
    ].head()
)