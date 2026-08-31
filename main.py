"""
LivingMemoryOS-CAMR (v2)
Clinical-Aware Memory Replacement for resource-constrained healthcare devices.

Changes from the original prototype are called out inline with "# FIX:"
comments so they're easy to diff against the original script.
"""

import heapq
import itertools

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG  (FIX: magic numbers pulled out and named so they
# can be justified/tuned/cited in a patent spec)
# =====================================================

RANDOM_STATE = 42

CARE_LEVEL_THRESHOLDS = {"ER": 0.45, "ICU": 0.35, "HDU": 0.20}
BED_PRIORITY = {"ER": 1.00, "ICU": 0.80, "HDU": 0.50, "WARD": 0.20}
ESCALATION_SCORE = {"ER": 1.00, "ICU": 0.75, "HDU": 0.50, "WARD": 0.25}

CMS_WEIGHTS = dict(
    criticality=0.30,
    severity=0.20,
    bed_priority=0.15,
    biomarker_risk=0.15,
    sepsis_risk=0.10,
    escalation_score=0.10,
)

# FIX: three separately-budgeted tiers instead of one flat 100-slot list.
# This is what your problem statement actually describes; the original
# code only *labeled* pages by tier inside a single pool, so nothing
# stopped Emergency pages from being crowded out or Normal from
# swallowing the whole budget.
TIER_CAPACITY = {"EMERGENCY": 15, "HIGH_PRIORITY": 35, "NORMAL": 50}

# FIX: this threshold used to be the 90th percentile of the *test set's*
# own criticality scores -- i.e. the cutoff was defined by the data it
# was then evaluated against. That's circular and won't generalize to a
# live device. Fix it as a clinically-motivated constant (tune this
# against a validation set, not the reported test set).
CRITICAL_THRESHOLD = 0.30

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("ICU_Patient_Monitoring_Mortality_Prediction_15000.csv")

features = [
    "heart_rate_mean", "spo2_mean", "respiratory_rate_mean",
    "temperature_mean", "apache_score", "sofa_score",
    "glucose_mean", "lactate_mean", "comorbidity_score", "sepsis_flag",
]

X = df[features]
y = df["mortality_label"]

# FIX: stratify=y. mortality_label is ~77/23 imbalanced; an unstratified
# split can shift that ratio in the test fold and makes accuracy alone
# misleading (see metrics below).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# FIX: keep the original dataframe rows (esp. patient_id) aligned to the
# test split so every memory page can be traced back to a real patient
# instead of a throwaway positional index.
test_meta = df.loc[X_test.index].reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# =====================================================
# MODEL
# =====================================================

base_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=RANDOM_STATE
)

# FIX: RandomForest's predict_proba is not a calibrated probability --
# treating it directly as a "mortality risk" that then drives clinical
# escalation (ER/ICU/HDU/WARD) is exactly the kind of claim a reviewer
# (or an examiner, or a hospital IRB) will push back on. Calibrate it.
model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# FIX: report metrics that mean something under class imbalance, not
# just accuracy (a model that always predicts "survives" would still
# score ~0.77 accuracy here).
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, predictions, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
print("PR-AUC :", round(average_precision_score(y_test, probs), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

criticality_scores = probs

# =====================================================
# CARE ESCALATION
# =====================================================

def assign_care_level(criticality: float) -> str:
    if criticality >= CARE_LEVEL_THRESHOLDS["ER"]:
        return "ER"
    if criticality >= CARE_LEVEL_THRESHOLDS["ICU"]:
        return "ICU"
    if criticality >= CARE_LEVEL_THRESHOLDS["HDU"]:
        return "HDU"
    return "WARD"


def build_page(idx: int) -> dict:
    row = X_test.iloc[idx]
    meta = test_meta.iloc[idx]
    criticality = float(criticality_scores[idx])
    care_level = assign_care_level(criticality)

    severity = min(
        ((row["apache_score"] / 40) + (row["sofa_score"] / 20)) / 2, 1.0
    )
    biomarker_risk = min(row["lactate_mean"] / 10, 1.0)
    sepsis_risk = float(row["sepsis_flag"])

    cms = (
        CMS_WEIGHTS["criticality"] * criticality
        + CMS_WEIGHTS["severity"] * severity
        + CMS_WEIGHTS["bed_priority"] * BED_PRIORITY[care_level]
        + CMS_WEIGHTS["biomarker_risk"] * biomarker_risk
        + CMS_WEIGHTS["sepsis_risk"] * sepsis_risk
        + CMS_WEIGHTS["escalation_score"] * ESCALATION_SCORE[care_level]
    )

    tier = "NORMAL"
    if criticality >= CRITICAL_THRESHOLD:
        tier = "HIGH_PRIORITY"
    if criticality >= CRITICAL_THRESHOLD and row["sepsis_flag"] == 1:
        tier = "EMERGENCY"

    reasons = []
    if criticality >= CRITICAL_THRESHOLD:
        reasons.append("High Mortality Risk")
    if row["sepsis_flag"] == 1:
        reasons.append("Sepsis")
    if row["lactate_mean"] > 4:
        reasons.append("High Lactate")
    if care_level == "ER":
        reasons.append("ER Escalation")
    if tier == "EMERGENCY":
        reasons.append("Emergency Preservation")
    if not reasons:
        reasons.append("Low Clinical Priority")

    return {
        "page_id": idx,
        "patient_id": meta["patient_id"],
        "criticality": round(criticality, 4),
        "care_level": care_level,
        "severity": round(float(severity), 4),
        "biomarker_risk": round(float(biomarker_risk), 4),
        "sepsis": int(row["sepsis_flag"]),
        "cms": round(float(cms), 4),
        "tier": tier,
        "reason": ", ".join(reasons),
    }


memory_pages = [build_page(i) for i in range(len(X_test))]

# =====================================================
# BASELINE: pure FIFO over a single 100-slot pool
# (kept from the original script as the "access-history-only"
# baseline the problem statement argues against)
# =====================================================

FIFO_SIZE = sum(TIER_CAPACITY.values())
fifo_memory = []
for page in memory_pages:
    if len(fifo_memory) >= FIFO_SIZE:
        fifo_memory.pop(0)
    fifo_memory.append(page)

# =====================================================
# LIVING MEMORY OS -- three independently bounded tiers,
# each a min-heap on CMS for O(log n) eviction instead of the
# O(n) full-list scan the original version did per admission.
# =====================================================

_counter = itertools.count()  # tie-breaker so heap never compares dicts


class TierPool:
    def __init__(self, capacity: int, evictable: bool = True):
        self.capacity = capacity
        self.evictable = evictable
        self.heap = []  # (cms, tie, page)
        self.by_id = {}

    def admit(self, page: dict) -> bool:
        if len(self.heap) < self.capacity:
            entry = (page["cms"], next(_counter), page)
            heapq.heappush(self.heap, entry)
            self.by_id[page["page_id"]] = page
            return True
        if not self.evictable:
            # Emergency tier: never silently evict. At true capacity we
            # reject/alert rather than displace another critical patient.
            return False
        weakest_cms, _, weakest_page = self.heap[0]
        if page["cms"] > weakest_cms:
            heapq.heapreplace(self.heap, (page["cms"], next(_counter), page))
            del self.by_id[weakest_page["page_id"]]
            self.by_id[page["page_id"]] = page
            return True
        return False

    def pages(self):
        return [p for _, _, p in self.heap]


tiers = {
    "EMERGENCY": TierPool(TIER_CAPACITY["EMERGENCY"], evictable=False),
    "HIGH_PRIORITY": TierPool(TIER_CAPACITY["HIGH_PRIORITY"]),
    "NORMAL": TierPool(TIER_CAPACITY["NORMAL"]),
}

rejected_emergency = 0
for page in memory_pages:
    admitted = tiers[page["tier"]].admit(page)
    if not admitted and page["tier"] == "EMERGENCY":
        rejected_emergency += 1  # capacity-full alert condition

living_memory = (
    tiers["EMERGENCY"].pages()
    + tiers["HIGH_PRIORITY"].pages()
    + tiers["NORMAL"].pages()
)

# =====================================================
# METRICS: fraction of clinically-critical pages retained
# =====================================================

def critical_fraction(pages):
    if not pages:
        return 0.0
    return sum(p["criticality"] >= CRITICAL_THRESHOLD for p in pages) / len(pages)


fifo_critical = sum(p["criticality"] >= CRITICAL_THRESHOLD for p in fifo_memory)
living_critical = sum(p["criticality"] >= CRITICAL_THRESHOLD for p in living_memory)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print("FIFO Critical Pages:", fifo_critical, f"/ {len(fifo_memory)}")
print("LivingMemoryOS Critical Pages:", living_critical, f"/ {len(living_memory)}")
print("Emergency pages rejected at full capacity:", rejected_emergency)

for name, pool in tiers.items():
    print(f"{name}: {len(pool.pages())}/{pool.capacity}")

if fifo_critical > 0:
    improvement = ((living_critical - fifo_critical) / fifo_critical) * 100
    print("\nImprovement over FIFO:", round(improvement, 2), "%")

# =====================================================
# EXPORT
# =====================================================

results_df = pd.DataFrame(living_memory)
results_df.to_csv("livingmemory_results_v2.csv", index=False)
print("\nSaved: livingmemory_results_v2.csv")