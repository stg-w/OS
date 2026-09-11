import pandas as pd

patients = pd.read_csv("patients_sample.csv")
admissions = pd.read_csv("admissions_sample.csv")
icu = pd.read_csv("icustays_sample.csv")
transfers = pd.read_csv("transfers_sample.csv")

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
    on=["subject_id","hadm_id"],
    how="left"
)
master.to_csv(
    "master.csv",
    index=False
)

print("master.csv created successfully")
print("Rows:", len(master))
print("Columns:", len(master.columns))
print(master.columns.tolist())