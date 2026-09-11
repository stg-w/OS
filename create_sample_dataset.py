import pandas as pd

pd.read_csv(
    "chartevents.csv",
    nrows=50000
).to_csv(
    "chartevents_sample.csv",
    index=False
)

pd.read_csv(
    "labevents.csv",
    nrows=50000
).to_csv(
    "labevents_sample.csv",
    index=False
)

pd.read_csv(
    "transfers.csv",
    nrows=50000
).to_csv(
    "transfers_sample.csv",
    index=False
)

pd.read_csv(
    "icustays.csv",
    nrows=50000
).to_csv(
    "icustays_sample.csv",
    index=False
)

pd.read_csv(
    "patients.csv"
).to_csv(
    "patients_sample.csv",
    index=False
)

pd.read_csv(
    "admissions.csv"
).to_csv(
    "admissions_sample.csv",
    index=False
)

print("Done")