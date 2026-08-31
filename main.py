import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "ICU_Patient_Monitoring_Mortality_Prediction_15000.csv"
)

# =====================================================
# FEATURES
# =====================================================

features = [
    "heart_rate_mean",
    "spo2_mean",
    "respiratory_rate_mean",
    "temperature_mean",
    "apache_score",
    "sofa_score",
    "glucose_mean",
    "lactate_mean",
    "comorbidity_score",
    "sepsis_flag"
]

X = df[features]
y = df["mortality_label"]

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# MODEL EVALUATION
# =====================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print("Accuracy:", round(accuracy, 4))

# =====================================================
# MORTALITY RISK / CRITICALITY
# =====================================================

probs = model.predict_proba(X_test)

criticality_scores = probs[:, 1]

print("\nMax Criticality:",
      round(criticality_scores.max(), 4))

print("Min Criticality:",
      round(criticality_scores.min(), 4))

print("Average Criticality:",
      round(criticality_scores.mean(), 4))

# =====================================================
# TOP 10% = CRITICAL
# =====================================================

critical_threshold = np.percentile(
    criticality_scores,
    90
)

print(
    "\nCritical Threshold:",
    round(critical_threshold, 4)
)

# =====================================================
# CARE ESCALATION ENGINE
# =====================================================

def assign_care_level(criticality):

    if criticality >= 0.45:
        return "ER"

    elif criticality >= 0.35:
        return "ICU"

    elif criticality >= 0.20:
        return "HDU"

    else:
        return "WARD"

# =====================================================
# BED PRIORITIES
# =====================================================

BED_PRIORITY = {
    "ER": 1.00,
    "ICU": 0.80,
    "HDU": 0.50,
    "WARD": 0.20
}

# =====================================================
# ESCALATION PRIORITY
# =====================================================

ESCALATION_SCORE = {
    "ER": 1.00,
    "ICU": 0.75,
    "HDU": 0.50,
    "WARD": 0.25
}

# =====================================================
# CREATE MEMORY PAGES
# =====================================================

memory_pages = []

for idx in range(len(X_test)):

    row = X_test.iloc[idx]

    criticality = float(
        criticality_scores[idx]
    )

    care_level = assign_care_level(
        criticality
    )

    bed_priority = BED_PRIORITY[
        care_level
    ]

    escalation_score = ESCALATION_SCORE[
        care_level
    ]

    # =================================
    # SEVERITY
    # =================================

    severity = (
        (row["apache_score"] / 40)
        +
        (row["sofa_score"] / 20)
    ) / 2

    severity = min(
        severity,
        1.0
    )

    # =================================
    # BIOMARKER RISK
    # =================================

    biomarker_risk = min(
        row["lactate_mean"] / 10,
        1.0
    )

    # =================================
    # SEPSIS RISK
    # =================================

    sepsis_risk = float(
        row["sepsis_flag"]
    )

    # =================================
    # CMS
    # =================================

    cms = (
        0.30 * criticality
        +
        0.20 * severity
        +
        0.15 * bed_priority
        +
        0.15 * biomarker_risk
        +
        0.10 * sepsis_risk
        +
        0.10 * escalation_score
    )

    # =================================
    # MEMORY TIERS
    # =================================

    tier = "NORMAL"

    if criticality >= critical_threshold:
        tier = "HIGH_PRIORITY"

    if (
        criticality >= critical_threshold
        and row["sepsis_flag"] == 1
    ):
        tier = "EMERGENCY"

    # =================================
    # EXPLAINABILITY
    # =================================

    reasons = []

    if criticality >= critical_threshold:
        reasons.append(
            "High Mortality Risk"
        )

    if row["sepsis_flag"] == 1:
        reasons.append(
            "Sepsis"
        )

    if row["lactate_mean"] > 4:
        reasons.append(
            "High Lactate"
        )

    if care_level == "ER":
        reasons.append(
            "ER Escalation"
        )

    if tier == "EMERGENCY":
        reasons.append(
            "Emergency Preservation"
        )

    if len(reasons) == 0:
        reasons.append(
            "Low Clinical Priority"
        )

    page = {

        "page_id":
        idx,

        "criticality":
        round(criticality, 4),

        "care_level":
        care_level,

        "severity":
        round(float(severity), 4),

        "biomarker_risk":
        round(float(biomarker_risk), 4),

        "sepsis":
        int(row["sepsis_flag"]),

        "cms":
        round(float(cms), 4),

        "tier":
        tier,

        "reason":
        ", ".join(reasons)
    }

    memory_pages.append(page)

# =====================================================
# FIFO MEMORY
# =====================================================

MEMORY_SIZE = 100

fifo_memory = []

for page in memory_pages:

    if len(fifo_memory) >= MEMORY_SIZE:
        fifo_memory.pop(0)

    fifo_memory.append(page)

# =====================================================
# LIVING MEMORY OS
# =====================================================

living_memory = []

emergency_memory = []

for page in memory_pages:

    if page["tier"] == "EMERGENCY":
        emergency_memory.append(page)

    if len(living_memory) < MEMORY_SIZE:

        living_memory.append(page)

    else:

        candidates = [

            p

            for p in living_memory

            if p["tier"] != "EMERGENCY"
        ]

        if len(candidates) == 0:
            continue

        victim = min(
            candidates,
            key=lambda x: x["cms"]
        )

        if page["cms"] > victim["cms"]:

            living_memory.remove(
                victim
            )

            living_memory.append(
                page
            )

# =====================================================
# FIFO METRICS
# =====================================================

fifo_critical = 0

for page in fifo_memory:

    if page["criticality"] > critical_threshold:
        fifo_critical += 1

# =====================================================
# LIVING MEMORY METRICS
# =====================================================

living_critical = 0

for page in living_memory:

    if page["criticality"] > critical_threshold:
        living_critical += 1

# =====================================================
# MEMORY TIER STATS
# =====================================================

normal_count = 0
high_count = 0
emergency_count = 0

for page in living_memory:

    if page["tier"] == "NORMAL":
        normal_count += 1

    elif page["tier"] == "HIGH_PRIORITY":
        high_count += 1

    elif page["tier"] == "EMERGENCY":
        emergency_count += 1

# =====================================================
# RESULTS
# =====================================================

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    "FIFO Critical Pages:",
    fifo_critical
)

print(
    "LivingMemoryOS Critical Pages:",
    living_critical
)

print(
    "\nMemory Tier Distribution"
)

print(
    "Normal:",
    normal_count
)

print(
    "High Priority:",
    high_count
)

print(
    "Emergency:",
    emergency_count
)

print(
    "Emergency Memory Pool:",
    len(emergency_memory)
)

if fifo_critical > 0:

    improvement = (
        (
            living_critical
            -
            fifo_critical
        )
        /
        fifo_critical
    ) * 100

    print(
        "\nImprovement:",
        round(improvement, 2),
        "%"
    )

# =====================================================
# TOP RETAINED PAGES
# =====================================================

top_pages = sorted(
    living_memory,
    key=lambda x: x["cms"],
    reverse=True
)

print("\n==============================")
print("TOP 10 RETAINED PAGES")
print("==============================")

for page in top_pages[:10]:

    print(
        f"Page:{page['page_id']} | "
        f"CMS:{page['cms']} | "
        f"Tier:{page['tier']} | "
        f"Care:{page['care_level']} | "
        f"{page['reason']}"
    )

# =====================================================
# EXPORT
# =====================================================

results_df = pd.DataFrame(
    living_memory
)

results_df.to_csv(
    "livingmemory_results.csv",
    index=False
)

print(
    "\nSaved: livingmemory_results.csv"
)