"""
LivingMemoryOS Configuration & Constants
"""

RANDOM_STATE = 42

DEFAULT_DATASET_PATH = "ICU_Patient_Monitoring_Mortality_Prediction_15000.csv"

FEATURES = [
    "heart_rate_mean",
    "spo2_mean",
    "respiratory_rate_mean",
    "temperature_mean",
    "apache_score",
    "sofa_score",
    "glucose_mean",
    "lactate_mean",
    "comorbidity_score",
    "sepsis_flag",
]

TARGET = "mortality_label"

DEFAULT_CARE_LEVEL_THRESHOLDS = {
    "ER": 0.45,
    "ICU": 0.35,
    "HDU": 0.20,
}

BED_PRIORITY = {
    "ER": 1.00,
    "ICU": 0.80,
    "HDU": 0.50,
    "WARD": 0.20,
}

ESCALATION_SCORE = {
    "ER": 1.00,
    "ICU": 0.75,
    "HDU": 0.50,
    "WARD": 0.25,
}

DEFAULT_CMS_WEIGHTS = {
    "criticality": 0.30,
    "severity": 0.20,
    "bed_priority": 0.15,
    "biomarker_risk": 0.15,
    "sepsis_risk": 0.10,
    "escalation_score": 0.10,
}

DEFAULT_TIER_CAPACITY = {
    "EMERGENCY": 15,
    "HIGH_PRIORITY": 35,
    "NORMAL": 50,
}

DEFAULT_CRITICAL_THRESHOLD = 0.30
