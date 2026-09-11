import pandas as pd

# =====================================================
# LOAD MODEL OUTPUT
# =====================================================

master = pd.read_csv("model_output.csv")

# =====================================================
# PATIENT-CRITICALITY INHERITANCE
# =====================================================

master["inheritance"] = (
    0.5 * master["icu_risk"]
    +
    0.5 * master["escalation_risk"]
)

# =====================================================
# CLINICAL MEMORY SCORE (CMS)
# =====================================================

master["cms"] = (
    0.30 * master["ai_risk"]
    +
    0.20 * master["biomarker_risk"]
    +
    0.20 * master["escalation_risk"]
    +
    0.15 * master["icu_risk"]
    +
    0.10 * master["age_risk"]
    +
    0.05 * master["inheritance"]
)

# =====================================================
# PROGNOSTIC RETENTION
# =====================================================

master["protected"] = (
    (master["ai_risk"] > 0.75)
    |
    (master["biomarker_risk"] > 0.80)
)

# =====================================================
# CREATE MEMORY PAGES
# =====================================================

pages = master.to_dict(orient="records")

MEMORY_SIZE = 100

# =====================================================
# FIFO BASELINE
# =====================================================

fifo = []

for page in pages:

    if len(fifo) >= MEMORY_SIZE:
        fifo.pop(0)

    fifo.append(page)

# =====================================================
# LIVING MEMORY OS
# =====================================================

living = []

for page in pages:

    if len(living) < MEMORY_SIZE:

        living.append(page)

    else:

        candidates = [
            p
            for p in living
            if not p["protected"]
        ]

        if len(candidates) == 0:
            continue

        victim = min(
            candidates,
            key=lambda x: x["cms"]
        )

        if page["cms"] > victim["cms"]:

            living.remove(victim)
            living.append(page)

# =====================================================
# SELF-EVOLVING CAPACITY
# =====================================================

emergency_pages = sum(
    1
    for p in living
    if p["cms"] > 0.80
)

ADAPTIVE_MEMORY_SIZE = MEMORY_SIZE

if emergency_pages > 30:
    ADAPTIVE_MEMORY_SIZE += 20

# =====================================================
# METRICS
# =====================================================

fifo_average = sum(
    p["cms"]
    for p in fifo
) / len(fifo)

living_average = sum(
    p["cms"]
    for p in living
) / len(living)

protected_pages = sum(
    1
    for p in living
    if p["protected"]
)

high_priority_pages = sum(
    1
    for p in living
    if p["cms"] > 0.60
)

# =====================================================
# RESULTS
# =====================================================

print("\n==============================")
print("LIVING MEMORY OS RESULTS")
print("==============================")

print("FIFO Average CMS:",
      round(fifo_average, 4))

print("LivingMemory Average CMS:",
      round(living_average, 4))

print("Protected Pages:",
      protected_pages)

print("High Priority Pages:",
      high_priority_pages)

print("Emergency Pages:",
      emergency_pages)

print("Adaptive Capacity:",
      ADAPTIVE_MEMORY_SIZE)

if fifo_average > 0:

    improvement = (
        (living_average - fifo_average)
        / fifo_average
    ) * 100

    print(
        "Retention Improvement:",
        round(improvement, 2),
        "%"
    )

# =====================================================
# SAVE RESULTS
# =====================================================

results = pd.DataFrame(living)

results.to_csv(
    "livingmemory_results_v4.csv",
    index=False
)

# =====================================================
# CAPACITY EVOLUTION FILE
# =====================================================

capacity_df = pd.DataFrame({
    "initial_capacity": [MEMORY_SIZE],
    "adaptive_capacity": [ADAPTIVE_MEMORY_SIZE],
    "emergency_pages": [emergency_pages],
    "protected_pages": [protected_pages]
})

capacity_df.to_csv(
    "capacity_evolution_v4.csv",
    index=False
)

print("\nSaved: livingmemory_results_v4.csv")
print("Saved: capacity_evolution_v4.csv")