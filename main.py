"""
Self-Evolving Clinical Memory Architecture with Prognostic Retention and
Patient-Criticality Inheritance for Healthcare Edge Devices 
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
# CONFIG
# =====================================================

RANDOM_STATE = 42

CARE_LEVEL_THRESHOLDS = {"ER": 0.45, "ICU": 0.35, "HDU": 0.20}
BED_PRIORITY = {"ER": 1.00, "ICU": 0.80, "HDU": 0.50, "WARD": 0.20}
ESCALATION_SCORE = {"ER": 1.00, "ICU": 0.75, "HDU": 0.50, "WARD": 0.25}

# NEW: prognostic_trend weight added; other weights rebalanced so the
# set still sums to 1.0
CMS_WEIGHTS = dict(
    criticality=0.25,
    severity=0.15,
    bed_priority=0.15,
    biomarker_risk=0.15,
    sepsis_risk=0.10,
    escalation_score=0.10,
    prognostic_trend=0.10,
)

CRITICAL_THRESHOLD = 0.30

# NEW: early-warning trigger. A patient below CRITICAL_THRESHOLD but
# showing high vital-sign instability still gets pulled up a tier.
# NOTE: set from the ~90th percentile of instability_index on this
# dataset (checked empirically, not guessed) -- re-derive this from your
# own training distribution if you swap datasets.
INSTABILITY_ESCALATION_THRESHOLD = 0.33
PROGNOSTIC_PRE_ALERT_CRITICALITY = CRITICAL_THRESHOLD * 0.7

# NEW: self-evolving capacity bounds. HIGH_PRIORITY stays fixed;
# EMERGENCY and NORMAL trade capacity with each other but the three
# tiers always sum to TOTAL_CAPACITY -- the fixed edge-device budget.
HIGH_PRIORITY_CAPACITY = 35
TOTAL_CAPACITY = 100
EMERGENCY_MIN, EMERGENCY_MAX = 10, 40
NORMAL_MIN = TOTAL_CAPACITY - HIGH_PRIORITY_CAPACITY - EMERGENCY_MAX  # keeps sum fixed
BATCH_SIZE = 250
REJECT_RATE_HIGH = 0.10   # grow EMERGENCY if rejection rate exceeds this
REJECT_RATE_LOW = 0.01    # shrink EMERGENCY back if rejection rate this calm
CAPACITY_STEP = 3

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# NEW: keep the instability-relevant columns alongside patient_id, aligned
# to the same test split
trend_cols = ["patient_id", "heart_rate_mean", "heart_rate_std",
              "heart_rate_max", "heart_rate_min",
              "systolic_bp_mean", "systolic_bp_std"]
test_meta = df.loc[X_test.index, trend_cols].reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# =====================================================
# MODEL
# =====================================================

base_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=RANDOM_STATE
)
model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

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


def instability_index(meta_row) -> float:
    """NEW: proxy for prognostic/trend risk from within-stay vital
    variability (coefficient of variation on HR and systolic BP,
    plus HR range relative to mean). Clipped to [0, 1]."""
    hr_cv = meta_row["heart_rate_std"] / max(meta_row["heart_rate_mean"], 1e-6)
    bp_cv = meta_row["systolic_bp_std"] / max(meta_row["systolic_bp_mean"], 1e-6)
    hr_range = (
        (meta_row["heart_rate_max"] - meta_row["heart_rate_min"])
        / max(meta_row["heart_rate_mean"], 1e-6)
    )
    raw = (hr_cv + bp_cv + hr_range) / 3
    return float(min(raw, 1.0))


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
    trend_risk = instability_index(meta)  # NEW

    cms = (
        CMS_WEIGHTS["criticality"] * criticality
        + CMS_WEIGHTS["severity"] * severity
        + CMS_WEIGHTS["bed_priority"] * BED_PRIORITY[care_level]
        + CMS_WEIGHTS["biomarker_risk"] * biomarker_risk
        + CMS_WEIGHTS["sepsis_risk"] * sepsis_risk
        + CMS_WEIGHTS["escalation_score"] * ESCALATION_SCORE[care_level]
        + CMS_WEIGHTS["prognostic_trend"] * trend_risk
    )

    tier = "NORMAL"
    if criticality >= CRITICAL_THRESHOLD:
        tier = "HIGH_PRIORITY"
    if criticality >= CRITICAL_THRESHOLD and row["sepsis_flag"] == 1:
        tier = "EMERGENCY"

    # NEW: prognostic early-warning escalation -- trending unstable but
    # not yet past the raw criticality cutoff still earns HIGH_PRIORITY
    prognostic_escalated = False
    if (
        tier == "NORMAL"
        and criticality >= PROGNOSTIC_PRE_ALERT_CRITICALITY
        and trend_risk >= INSTABILITY_ESCALATION_THRESHOLD
    ):
        tier = "HIGH_PRIORITY"
        prognostic_escalated = True

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
    if prognostic_escalated:
        reasons.append("Prognostic Early Warning")
    if not reasons:
        reasons.append("Low Clinical Priority")

    return {
        "page_id": idx,
        "patient_id": meta["patient_id"],
        "criticality": round(criticality, 4),
        "care_level": care_level,
        "severity": round(float(severity), 4),
        "biomarker_risk": round(float(biomarker_risk), 4),
        "trend_risk": round(trend_risk, 4),
        "sepsis": int(row["sepsis_flag"]),
        "cms": round(float(cms), 4),
        "tier": tier,
        "reason": ", ".join(reasons),
    }


memory_pages = [build_page(i) for i in range(len(X_test))]

# =====================================================
# BASELINE: pure FIFO over a single 100-slot pool
# =====================================================

FIFO_SIZE = TOTAL_CAPACITY
fifo_memory = []
for page in memory_pages:
    if len(fifo_memory) >= FIFO_SIZE:
        fifo_memory.pop(0)
    fifo_memory.append(page)

# =====================================================
# LIVING MEMORY OS -- tiered heaps + self-evolving capacity
# =====================================================

_counter = itertools.count()


class TierPool:
    def __init__(self, capacity: int, evictable: bool = True):
        self.capacity = capacity
        self.evictable = evictable
        self.heap = []
        self.by_id = {}

    def admit(self, page: dict) -> bool:
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (page["cms"], next(_counter), page))
            self.by_id[page["page_id"]] = page
            return True
        if not self.evictable:
            return False
        weakest_cms, _, weakest_page = self.heap[0]
        if page["cms"] > weakest_cms:
            heapq.heapreplace(self.heap, (page["cms"], next(_counter), page))
            del self.by_id[weakest_page["page_id"]]
            self.by_id[page["page_id"]] = page
            return True
        return False

    def resize(self, new_capacity: int):
        """NEW: shrink/grow capacity at runtime. Evictable tiers give up
        their weakest pages if shrunk below current occupancy;
        non-evictable tiers just stop accepting new pages until their
        occupancy is back under the new cap (nothing already admitted is
        ever forced out)."""
        self.capacity = new_capacity
        if self.evictable:
            while len(self.heap) > self.capacity:
                cms, _, page = heapq.heappop(self.heap)
                del self.by_id[page["page_id"]]

    def pages(self):
        return [p for _, _, p in self.heap]


class MemoryController:
    """NEW: self-evolving allocator. Watches EMERGENCY rejection rate per
    batch and shifts capacity between EMERGENCY and NORMAL, keeping the
    three-tier total fixed at TOTAL_CAPACITY."""

    def __init__(self, tiers: dict):
        self.tiers = tiers
        self.history = []

    def observe_batch(self, rejected: int, batch_len: int):
        rate = rejected / batch_len if batch_len else 0.0
        emergency = self.tiers["EMERGENCY"]
        normal = self.tiers["NORMAL"]

        if rate > REJECT_RATE_HIGH and emergency.capacity < EMERGENCY_MAX:
            shift = min(CAPACITY_STEP, EMERGENCY_MAX - emergency.capacity,
                        normal.capacity - NORMAL_MIN)
            if shift > 0:
                emergency.resize(emergency.capacity + shift)
                normal.resize(normal.capacity - shift)
        elif rate < REJECT_RATE_LOW and emergency.capacity > EMERGENCY_MIN:
            shift = min(CAPACITY_STEP, emergency.capacity - EMERGENCY_MIN)
            if shift > 0:
                emergency.resize(emergency.capacity - shift)
                normal.resize(normal.capacity + shift)

        self.history.append(
            {"batch_reject_rate": round(rate, 3),
             "emergency_capacity": emergency.capacity,
             "normal_capacity": normal.capacity}
        )


tiers = {
    "EMERGENCY": TierPool(EMERGENCY_MIN, evictable=False),
    "HIGH_PRIORITY": TierPool(HIGH_PRIORITY_CAPACITY),
    "NORMAL": TierPool(TOTAL_CAPACITY - EMERGENCY_MIN - HIGH_PRIORITY_CAPACITY),
}
controller = MemoryController(tiers)

rejected_emergency_total = 0
batch_rejected = 0
for i, page in enumerate(memory_pages, start=1):
    admitted = tiers[page["tier"]].admit(page)
    if not admitted and page["tier"] == "EMERGENCY":
        rejected_emergency_total += 1
        batch_rejected += 1
    if i % BATCH_SIZE == 0:
        controller.observe_batch(batch_rejected, BATCH_SIZE)
        batch_rejected = 0

living_memory = (
    tiers["EMERGENCY"].pages() + tiers["HIGH_PRIORITY"].pages() + tiers["NORMAL"].pages()
)

# =====================================================
# RESULTS
# =====================================================

fifo_critical = sum(p["criticality"] >= CRITICAL_THRESHOLD for p in fifo_memory)
living_critical = sum(p["criticality"] >= CRITICAL_THRESHOLD for p in living_memory)
prognostic_saves = sum(p["reason"].find("Prognostic Early Warning") >= 0 for p in living_memory)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print("FIFO Critical Pages:", fifo_critical, f"/ {len(fifo_memory)}")
print("LivingMemoryOS Critical Pages:", living_critical, f"/ {len(living_memory)}")
print("Emergency pages rejected (final capacity):", rejected_emergency_total)
print("Retained via prognostic early warning:", prognostic_saves)

for name, pool in tiers.items():
    print(f"{name}: {len(pool.pages())}/{pool.capacity}")

if fifo_critical > 0:
    improvement = ((living_critical - fifo_critical) / fifo_critical) * 100
    print("\nImprovement over FIFO:", round(improvement, 2), "%")

print("\nCapacity evolution over batches (EMERGENCY / NORMAL):")
for h in controller.history:
    print(f"  reject_rate={h['batch_reject_rate']:<6} "
          f"EMERGENCY={h['emergency_capacity']:<3} NORMAL={h['normal_capacity']}")

# =====================================================
# EXPORT
# =====================================================

results_df = pd.DataFrame(living_memory)
results_df.to_csv("livingmemory_results_v3.csv", index=False)
pd.DataFrame(controller.history).to_csv("capacity_evolution_v3.csv", index=False)
print("\nSaved: livingmemory_results_v3.csv, capacity_evolution_v3.csv")