"""
LivingMemoryOS v4 Configuration & Constants (MIMIC-IV Clinical Dataset)
"""

RANDOM_STATE = 42

# Dataset Paths
DATASET_PATH = "features.csv"
MODEL_OUTPUT_PATH = "model_output.csv"
RESULTS_V4_PATH = "livingmemory_results_v4.csv"
CAPACITY_V4_PATH = "capacity_evolution_v4.csv"

# Model Features & Target
FEATURES = [
    "age_risk",
    "icu_risk",
    "escalation_risk",
    "biomarker_risk",
]

TARGET = "hospital_expire_flag"

# Default Clinical Memory Score (CMS) Weights
DEFAULT_CMS_WEIGHTS = {
    "ai_risk": 0.30,
    "biomarker_risk": 0.20,
    "escalation_risk": 0.20,
    "icu_risk": 0.15,
    "age_risk": 0.10,
    "inheritance": 0.05,
}

# Prognostic Retention Thresholds
DEFAULT_PROTECTED_THRESHOLDS = {
    "ai_risk_cutoff": 0.75,
    "biomarker_risk_cutoff": 0.80,
}

# Edge Memory Capacity & Self-Evolving Rules
DEFAULT_CAPACITY = 100
ADAPTIVE_EXPANSION_STEP = 20
EMERGENCY_CMS_THRESHOLD = 0.80
ADAPTIVE_TRIGGER_COUNT = 30

