"""
MIMIC-IV Data Sampling and Preparation Utility

This script creates lightweight sample datasets from the original MIMIC-IV
tables to support faster development, testing, experimentation, and validation
of the LivingMemoryOS framework. It extracts manageable subsets from large
clinical event, laboratory, transfer, and ICU stay records while preserving
the original data structure and relationships required for feature engineering,
risk modeling, and memory simulation workflows. The generated sample files
provide a reproducible working dataset that enables rapid prototyping and
evaluation without requiring the full-scale MIMIC-IV database during routine
development and benchmarking tasks.
"""

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