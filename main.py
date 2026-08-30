import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "ICU_Patient_Monitoring_Mortality_Prediction_15000.csv"
)

# =========================
# FEATURES
# =========================

features = [
    "heart_rate_mean",
    "spo2_mean",
    "respiratory_rate_mean",
    "temperature_mean",
    "apache_score",
    "sofa_score"
]

X = df[features]
y = df["mortality_label"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATE MODEL
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

# =========================
# CRITICALITY SCORES
# =========================

probs = model.predict_proba(X_test)

criticality_scores = probs[:, 1]

print("\nFirst 5 Criticality Scores:")

for i in range(5):
    print(criticality_scores[i])

print("\nMax Criticality:",
      criticality_scores.max())

print("Min Criticality:",
      criticality_scores.min())

print("Average Criticality:",
      criticality_scores.mean())

# =========================
# TOP 10% = CRITICAL
# =========================

threshold = np.percentile(
    criticality_scores,
    90
)

print("\nCritical Threshold:", threshold)

# =========================
# CREATE MEMORY PAGES
# =========================

memory_pages = []

for i in range(len(X_test)):

    criticality = criticality_scores[i]

    cms = criticality

    page = {
        "page_id": i,
        "criticality": criticality,
        "cms": cms
    }

    memory_pages.append(page)

# =========================
# FIFO MEMORY
# =========================

MEMORY_SIZE = 100

fifo_memory = []

for page in memory_pages:

    if len(fifo_memory) >= MEMORY_SIZE:
        fifo_memory.pop(0)

    fifo_memory.append(page)

# =========================
# LIVING MEMORY OS
# =========================

living_memory = []

for page in memory_pages:

    if len(living_memory) < MEMORY_SIZE:

        living_memory.append(page)

    else:

        victim = min(
            living_memory,
            key=lambda x: x["cms"]
        )

        if page["cms"] > victim["cms"]:

            living_memory.remove(victim)

            living_memory.append(page)

# =========================
# COUNT FIFO CRITICAL PAGES
# =========================

fifo_critical = 0

for page in fifo_memory:

    if page["criticality"] > threshold:
        fifo_critical += 1

# =========================
# COUNT LIVINGMEMORYOS
# =========================

living_critical = 0

for page in living_memory:

    if page["criticality"] > threshold:
        living_critical += 1

# =========================
# RESULTS
# =========================

print("\n===================")
print("RESULTS")
print("===================")

print(
    "FIFO Critical Pages:",
    fifo_critical
)

print(
    "LivingMemoryOS Critical Pages:",
    living_critical
)

if fifo_critical > 0:

    improvement = (
        (living_critical - fifo_critical)
        / fifo_critical
    ) * 100

    print(
        "Improvement:",
        round(improvement, 2),
        "%"
    )