Clinical data
    ↓
Data preparation
    ↓
Risk-feature generation
    ↓
Mortality-risk prediction
    ↓
Clinical Memory Score calculation
    ↓
Protection and tier assignment
    ↓
FIFO and LivingMemoryOS simulation
    ↓
Adaptive capacity adjustment
    ↓
Performance comparison
    ↓
Dashboard/API presentation
```

The implementation is divided mainly between `mimic_simulator.py`, `mimic_classifier.py`, `mimic_scoring.py`, and `dashboard_new.py`.

---              

## 1. Input clinical data

The system uses MIMIC-IV-derived clinical data.

The main source tables are:

- `patients.csv`
- `admissions.csv`
- `icustays.csv`
- `transfers.csv`
- `labevents.csv`
- `chartevents.csv`

The system combines patient, admission, ICU, transfer, and laboratory information into a structured record.

The resulting engineered dataset is `features.csv`.

The current feature table contains approximately 550,818 records.

---

## 2. Feature engineering

The project converts raw medical information into four normalized risk features.

The feature-engineering logic is in `risk_features.py`.

### Age risk

```text
age_risk = min(anchor_age / 100, 1)
```

This converts age into a normalized value between 0 and 1.

### ICU risk

```text
icu_risk = min(length_of_stay / 10, 1)
```

A longer ICU stay produces a higher ICU-risk score.

### Escalation risk

The system counts patient transfers:

```text
transfer_count = number of transfer records
```

Then:

```text
escalation_risk = min(transfer_count / 10, 1)
```

Repeated transfers are treated as an indicator of clinical instability or escalation.

### Biomarker risk

Laboratory results with abnormal flags are counted:

```text
abnormal_lab = 1 if the laboratory flag is present
abnormal_lab = 0 otherwise
```

Then:

```text
biomarker_risk = min(abnormal_lab_count / 20, 1)
```

The four model features are configured in `config_new.py`:

```text
age_risk
icu_risk
escalation_risk
biomarker_risk
```

The target label is:

```text
hospital_expire_flag
```

This represents in-hospital mortality.

---

## 3. Mortality-risk prediction

The classifier is implemented in `mimic_classifier.py`.

The system uses a Random Forest classifier:

```text
Number of trees: 100
Maximum depth: 10
Random seed: 42
```

The data is split into:

```text
80% training data
20% testing data
```

The split is stratified so that the mortality distribution is maintained in both subsets.

If the dataset contains more than 150,000 rows, the classifier samples 150,000 rows for training and evaluation.

The classifier produces:

```text
AI risk = predicted probability of in-hospital mortality
```

The classifier also calculates:

- Accuracy.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.
- Classification report.
- Feature importance.

The trained model is cached in:

`mimic_model.joblib`

This prevents the application from retraining the model every time the dashboard opens.

---

## 4. Converting a patient into a memory page

The simulator treats every patient record as a memory page.

This happens in `mimic_simulator.py`.

For every input record, the simulator creates or adds:

```text
page_id
subject_id
hadm_id
age_risk
icu_risk
escalation_risk
biomarker_risk
ai_risk
inheritance
cms
protected
tier
reason
```

The `page_id` identifies the order in which the page entered the simulation stream.

The page is then evaluated by the scoring engine.

---

## 5. Patient-Criticality Inheritance

The inheritance score is calculated in `mimic_scoring.py`.

```text
inheritance =
    0.5 × icu_risk
  + 0.5 × escalation_risk
```

The purpose of inheritance is to represent continuing clinical importance.

A patient with:

- Prolonged ICU exposure.
- Frequent transfers.
- Repeated escalation.

receives a higher inheritance value.

The inheritance score contributes 5% to the final CMS.

---

## 6. Clinical Memory Score

The Clinical Memory Score is the central prioritization value.

It is calculated as:

```text
CMS =
    0.30 × AI mortality risk
  + 0.20 × biomarker risk
  + 0.20 × escalation risk
  + 0.15 × ICU risk
  + 0.10 × age risk
  + 0.05 × inheritance
```

The weights are defined in `config_new.py`.

The CMS represents the estimated importance of keeping a patient record in limited memory.

A higher CMS means that the record has higher retention priority.

The CMS is rounded to four decimal places.

---

## 7. Prognostic protection

The system marks a page as protected when either of the following conditions is satisfied:

```text
AI risk > 0.75
OR
biomarker risk > 0.80
```

The logic is implemented in `mimic_scoring.py`.

A protected page is intended to represent a clinically important record that should not be removed from memory under normal replacement conditions.

The protection result is stored as:

```text
protected = True
```

Important distinction:

- Protection is based on AI risk or biomarker risk.
- Emergency tier assignment can also occur from high CMS.
- Therefore, an `EMERGENCY` page is not necessarily protected unless it crosses one of the protection thresholds.

---

## 8. Tier assignment

The system assigns each page to one of three tiers.

The logic is:

```text
If protected is True:
    tier = EMERGENCY

Else if CMS >= 0.80:
    tier = EMERGENCY

Else if CMS >= 0.60:
    tier = HIGH_PRIORITY

Else:
    tier = NORMAL
```

### EMERGENCY

Represents the highest-priority records.

A page enters this tier when:

- It is prognostically protected, or
- Its CMS is at least 0.80.

### HIGH_PRIORITY

A page enters this tier when:

```text
0.60 <= CMS < 0.80
```

unless it is already protected.

### NORMAL

A page enters this tier when:

```text
CMS < 0.60
```

and it is not protected.

The system also creates a human-readable reason, such as:

- Prognostic Non-Evictable Protection.
- Critical AI Mortality Risk.
- Severe Lab Biomarker Abnormalities.
- High Transfer Escalation.
- Prolonged ICU Stay.
- Geriatric High Risk.
- Routine Inpatient Care.

---

# Detailed simulator workflow

The main simulator is the class:

```python
MimicLivingMemorySimulator
```

in `mimic_simulator.py`.

---

## 9. Simulator initialization

When the simulator is created:

1. A classifier is accepted or created.
2. If the classifier is not trained, it loads the cached model or trains a new model.
3. A clinical scorer is accepted or created.

The constructor is:

```python
MimicLivingMemorySimulator(
    classifier=None,
    scorer=None
)
```

The simulator is used by:

- `main_new.py`
- `dashboard_new.py`
- `server_new.py`

---

## 10. Stream loading

The method is:

```python
load_stream_pages(max_pages=10000)
```

The simulator chooses the input file in this order:

1. `model_output.csv`, if it exists.
2. Otherwise `features.csv`.

The stream is limited to the requested number of pages.

For example:

- The dashboard defaults to 5,000 pages.
- The command-line benchmark uses 10,000 pages.
- The API defaults to 5,000 pages.

If the input file does not already contain `ai_risk`, the simulator uses the trained classifier to calculate it.

Missing risk values are filled with zero.

The resulting rows are converted into a list of Python dictionaries.

---

## 11. Scoring every page

For each page in the stream, the simulator calls:

```python
scorer.evaluate_patient(...)
```

The scorer calculates:

1. Inheritance.
2. CMS.
3. Protection status.
4. Tier.
5. Clinical reason.

The simulator then adds those values to the page dictionary.

At this point, every page has both its original patient information and its memory-management metadata.

---

## 12. FIFO baseline

The first comparison strategy is FIFO.

FIFO means First-In-First-Out.

The FIFO algorithm works as follows:

```text
For each incoming page:
    If memory is full:
        remove the oldest page
    Add the incoming page
```

FIFO does not consider clinical importance.

It does not use:

- AI mortality risk.
- CMS.
- Protection.
- ICU risk.
- Biomarker risk.
- Escalation risk.

Its only rule is arrival order.

The FIFO memory remains limited to the original memory size.

---

## 13. LivingMemoryOS retention

The second strategy is the LivingMemoryOS retention algorithm.

The memory starts empty.

For each incoming page:

### Case 1: Memory has available space

The page is admitted immediately.

### Case 2: Memory is full

The simulator finds all retained pages that are not protected:

```text
candidate pages =
    retained pages where protected == False
```

### Case 3: All existing pages are protected

If no unprotected page exists:

```text
the incoming page is rejected
```

This prevents a protected page from being displaced.

### Case 4: Unprotected pages exist

The simulator selects:

```text
the unprotected page with the lowest CMS
```

This page is the eviction candidate.

The incoming page replaces the candidate only when:

```text
incoming CMS > victim CMS
```

If the incoming CMS is not higher, the incoming page is rejected.

This means the algorithm is not simply “keep every high-risk page.” It performs a comparative admission decision between the new page and the weakest currently retained unprotected page.

---

## 14. Adaptive capacity

The simulator also monitors emergency-page load.

The defaults are defined in `config_new.py`:

```text
Initial capacity: 100
Expansion step: 20
Emergency threshold: 0.80
Trigger count: 30
```

After each page is processed, the simulator counts retained pages satisfying:

```text
CMS >= emergency_threshold
```

When the count reaches the next trigger, the capacity increases:

```text
adaptive_capacity += adaptive_expansion_step
```

The trigger is then moved forward by another trigger block.

This prevents repeated expansion on every page.

For example:

```text
Initial capacity = 100
Trigger count = 30
Expansion step = 20
```

When 30 emergency-level pages are retained:

```text
capacity becomes 120
```

When the next emergency block is reached:

```text
capacity becomes 140
```

The expanded capacity is used for subsequent page admissions.

The result contains both:

```text
memory_size
adaptive_capacity
```

This is a simulation of adaptive capacity. It does not directly allocate computer RAM or storage hardware.

---

## 15. Simulation metrics

After processing the stream, the simulator calculates:

### FIFO average CMS

```text
average CMS of pages retained by FIFO
```

### LivingMemory average CMS

```text
average CMS of pages retained by LivingMemoryOS
```

### Retention improvement

```text
((LivingMemory average CMS - FIFO average CMS)
 / FIFO average CMS) × 100
```

### Protected pages

The number of retained records where:

```text
protected == True
```

### FIFO protected pages

The number of protected pages remaining in the FIFO buffer.

### High-priority pages

The number of retained pages where:

```text
CMS >= 0.60
```

### Emergency pages

The number of retained pages where:

```text
CMS >= emergency threshold
```

### Rejected count

The number of incoming pages rejected because:

- All available pages were protected, or
- The new page had lower or equal CMS than the selected victim.

### Eviction count

The number of successful replacements of an existing unprotected page.

Finally, LivingMemoryOS pages are sorted in descending order by CMS.

---

# Website dashboard workflow

The main dashboard is `dashboard_new.py`.

It is started directly through `app_new.py` or selected through `app.py`.

The dashboard is built using:

- Streamlit.
- pandas.
- NumPy.
- Plotly.
- The MIMIC-IV classifier.
- The MIMIC-IV simulator.

---

## 16. Dashboard initialization

The dashboard calls:

```python
get_simulator()
```

This function is decorated with:

```python
@st.cache_resource
```

Therefore, Streamlit attempts to reuse the initialized model and simulator rather than rebuilding them on every interaction.

The initialization sequence is:

```text
Create MimicMortalityClassifier
    ↓
Load cached model or train model
    ↓
Create MimicLivingMemorySimulator
    ↓
Return cached simulator
```

If initialization fails, the dashboard displays an error and stops rendering.

---

## 17. Dashboard visual design

The dashboard uses a medical-style user interface.

The visual elements include:

- White background.
- Royal-blue accents.
- Coral emergency indicators.
- Rounded feature cards.
- Hero banner.
- Sidebar controls.
- Animated content containers.
- Plotly charts.
- Streamlit tabs.
- Download buttons.
- Filterable tables.

The dashboard title identifies the system as:

```text
LivingMemoryOS v4
```

and describes it as a clinical admissions and ICU memory architecture.

The hero section presents:

- Intelligent memory allocation.
- MIMIC-IV inpatients.
- Patient-Criticality Inheritance.
- Prognostic Retention Lock.
- Self-Evolving Buffer Expansion.

These are explanatory UI labels for the implemented algorithms.

---

# Website dashboard components

## 18. Dataset selector in `app.py`

The outer application `app.py` displays a sidebar dataset selector.

The options are:

```text
MIMIC-IV Clinical Cohort
ICU Telemetry
```

When MIMIC-IV is selected, it imports:

```text
src.ui.dashboard_new
```

When ICU Telemetry is selected, it imports:

```text
src.ui.dashboard
```

The default selection is MIMIC-IV.

The dedicated `app_new.py` skips this selector and directly launches the MIMIC-IV dashboard.

---

## 19. System-control sidebar

The MIMIC-IV dashboard sidebar provides the main simulation configuration.

### Total Memory Capacity

The control is:

```text
20 to 300 slots
```

The default is:

```text
100 slots
```

This value becomes the simulator’s `memory_size`.

### Telemetry Stream Ingestion Size

The available values are:

```text
1000
2500
5000
10000
20000
```

This becomes the simulator’s `stream_size`.

### Active Memory Budget card

The sidebar displays the selected memory size in a visual information card.

### AI Mortality Cutoff

The allowed range is:

```text
0.50 to 0.95
```

The default is:

```text
0.75
```

This controls the protection threshold for AI risk.

### Biomarker Risk Cutoff

The allowed range is:

```text
0.50 to 0.95
```

The default is:

```text
0.80
```

This controls the protection threshold for biomarker risk.

---

## 20. CMS weight controls

The dashboard has a collapsible `CMS Clinical Weights` section.

The user can change:

- AI Risk Weight.
- Biomarker Risk Weight.
- Escalation Risk Weight.
- ICU Risk Weight.
- Age Risk Weight.
- Inheritance Weight.

Each value can range from 0.0 to 1.0.

The dashboard passes these values into:

```python
simulator.run_simulation(
    cms_weights=cms_weights
)
```

The dashboard does not enforce that the six weights sum to 1.0.

---

## 21. Self-evolving capacity controls

The dashboard has a collapsible `Self-Evolving Capacity Rules` section.

The user can change:

### Emergency CMS Threshold

Range:

```text
0.60 to 0.95
```

Default:

```text
0.80
```

### Trigger Emergency Page Count

Range:

```text
10 to 50
```

Default:

```text
30
```

### Capacity Expansion Step

Range:

```text
10 to 50
```

Default:

```text
20
```

These values are passed to the simulator and influence adaptive capacity.

---

# Dashboard tab 1: Memory Simulation and Comparative Analytics

The first tab is:

```text
Memory Simulation & Comparative Analytics
```

It calls:

```python
simulator.run_simulation(...)
```

The simulation runs whenever Streamlit reruns the page after a control change.

---

## 22. Summary statistics banner

The top statistics banner displays four values.

### LivingMemory retained CMS

This is:

```text
living_average_cms
```

It represents the average CMS of retained LivingMemoryOS pages.

### FIFO baseline CMS

This is:

```text
fifo_average_cms
```

It represents the average CMS retained by FIFO.

### Retention improvement

This is:

```text
improvement_pct
```

It compares LivingMemoryOS with FIFO.

### Protected patients

This displays:

```text
protected_pages / memory_size
```

The label describes these as non-evictable emergency pages.

Technically, the numerator counts pages with `protected == True`, while emergency tier membership may also arise from CMS alone.

---

## 23. Critical telemetry retention comparison chart

The dashboard creates a Plotly bar chart comparing:

- FIFO Baseline.
- LivingMemoryOS v4.

It displays protected-page retention values.

The chart is intended to show whether the clinical-priority strategy retains more protected records than FIFO.

This chart is a comparative visualization of simulation output, not a clinical outcome chart.

---

## 24. Tier pool allocation chart

The dashboard displays a stacked bar chart containing:

- Emergency.
- High Priority.
- Normal.

For each tier it displays:

- Occupied pages.
- Available headroom.

The displayed tier counts come from the retained page records.

Important implementation detail: the current MIMIC-IV simulator uses one global memory list and stores the tier as a label. The dashboard displays each tier with the selected total memory size as its “maximum capacity.” Therefore, this chart is a visualization of tier labels and counts, not three independent physical memory pools.

The actual separate heap-based tier pool implementation exists in the legacy `tier_pool.py`.

---

## 25. Active in-memory telemetry registry

The dashboard displays the retained LivingMemoryOS pages in a table.

The table can be filtered by:

### Memory tier

Options:

```text
EMERGENCY
HIGH_PRIORITY
NORMAL
```

### Care unit

The dashboard looks for:

```text
first_careunit
```

or, for legacy data:

```text
care_level
```

### Patient identifier

A text field filters the patient identifier.

The preferred displayed columns include:

- `page_id`
- `subject_id`
- `hadm_id`
- `admission_type`
- `first_careunit`
- `anchor_age`
- `los`
- `transfer_count`
- `abnormal`
- `ai_risk`
- `inheritance`
- `cms`
- `protected`
- `tier`
- `reason`

The table is sorted by CMS in descending order.

The dashboard also provides a CSV download button for the filtered table.

---

# Dashboard tab 2: Real-Time Patient Telemetry Triage

The second tab is:

```text
Real-Time Patient Telemetry Triage
```

It provides a simulated live-patient input form.

The form is displayed in three groups.

---

## 26. Demographics and stay inputs

The dashboard provides:

- Patient age.
- ICU length of stay.

The ranges are:

```text
Age: 18 to 100 years
ICU stay: 0 to 30 days
```

---

## 27. Acuity and biomarker inputs

The dashboard provides:

- Unit transfer count.
- Abnormal lab-event count.

The ranges are:

```text
Transfers: 0 to 20
Abnormal lab events: 0 to 200
```

---

## 28. Patient metadata inputs

The dashboard provides:

- Patient identifier.
- Admission type.
- Care unit.

Admission type choices include:

- `EMERGENCY`
- `EW EMER.`
- `DIRECT EMER.`
- `URGENT`
- `OBSERVATION ADMIT`
- `SURGICAL SAME DAY`

Care-unit choices include:

- `MICU`
- `SICU`
- `CCU`
- `CVICU`
- `TSICU`
- `WARD`

---

## 29. Preset clinical profiles

The dashboard provides preset examples.

### Sepsis Shock Crisis

Represents an elderly, highly unstable ICU patient with:

- High age.
- Long stay.
- Multiple transfers.
- Many abnormal laboratory events.
- MICU placement.

### Post-Surgical High-Risk ICU Patient

Represents:

- Older age.
- Prolonged ICU stay.
- Several transfers.
- High abnormal-lab count.
- Cardiovascular ICU placement.

### Stable Step-Down Ward Patient

Represents:

- Younger age.
- Short stay.
- No transfers.
- Few abnormal laboratory events.
- Ward placement.

These are demonstration templates. They are not clinical protocols.

---

## 30. Triage result cards

The dashboard displays four result cards.

### Mortality Risk

Shows:

```text
ai_risk × 100
```

This is displayed as a percentage.

### Inheritance

Shows:

```text
inheritance
```

The dashboard describes it as:

```text
0.5 × ICU + 0.5 × Escalation
```

### CMS Score

Shows:

```text
cms
```

This represents the Clinical Memory Index.

### Assigned Tier

Shows:

- `EMERGENCY`
- `HIGH_PRIORITY`
- `NORMAL`

It also displays:

- `NON-EVICTABLE` when protected.
- `EVICTABLE` otherwise.

---

## 31. Clinical preservation rationale

The dashboard displays the scorer’s `reason` string.

Example reasons may include:

```text
Prognostic Non-Evictable Protection
Critical AI Mortality Risk
Severe Lab Biomarker Abnormalities
High Transfer Escalation
Prolonged ICU Stay
Geriatric High Risk
```

This makes the algorithm’s decision more interpretable than returning only a numerical score.

---

# Dashboard tab 3: Clinical ML Model Validation

The third tab is:

```text
Clinical ML Model Validation
```

It presents the trained classifier’s evaluation information.

---

## 32. Accuracy card

Displays the classifier’s accuracy as a percentage.

```text
accuracy × 100
```

## 33. ROC-AUC card

Displays the receiver-operating-characteristic area under the curve.

This measures ranking/discrimination performance across classification thresholds.

## 34. PR-AUC card

Displays average precision / precision-recall area.

This is especially relevant for imbalanced mortality labels.

## 35. Feature-importance chart

The dashboard reads the Random Forest feature importances and displays them as a horizontal bar chart.

The displayed features are:

- `age_risk`
- `icu_risk`
- `escalation_risk`
- `biomarker_risk`

The feature importance is the model’s internal tree-based importance, not a causal medical interpretation.

## 36. Confusion matrix

The dashboard displays a matrix with:

```text
Predicted survival
Predicted mortality
Actual survival
Actual mortality
```

This visualizes:

- True negatives.
- False positives.
- False negatives.
- True positives.

## 37. Detailed classification report

The stored classification report is converted into a pandas DataFrame and displayed.

It can include:

- Precision.
- Recall.
- F1-score.
- Support.

---

# Exact explanation to use in a presentation

You can say:

> LivingMemoryOS is a clinical-priority memory-management framework. It receives patient records derived from MIMIC-IV, converts demographic, ICU, transfer, and laboratory information into normalized risk indicators, and uses those indicators with a Random Forest classifier to estimate in-hospital mortality risk.  
>
> Each patient record is then represented as a memory page. The system calculates Patient-Criticality Inheritance from ICU risk and escalation risk. It combines the AI risk, biomarker risk, escalation risk, ICU risk, age risk, and inheritance into a weighted Clinical Memory Score.  
>
> Based on AI risk, biomarker risk, and total CMS, each page receives a protection status and a memory tier. Protected pages are treated as non-evictable. During simulation, the system compares a conventional FIFO policy with LivingMemoryOS. FIFO removes the oldest page, whereas LivingMemoryOS identifies the lowest-CMS unprotected page and replaces it only when the incoming page has greater clinical memory value.  
>
> The system also monitors the number of retained emergency-level pages. When the emergency load reaches a configured threshold, the simulated memory capacity expands by a configured step. The final output includes average retained CMS, protected-page retention, emergency-page count, rejection count, eviction count, adaptive capacity, and the final retained memory state.  
>
> The Streamlit dashboard provides controls for memory capacity, stream size, protection thresholds, CMS weights, emergency threshold, and expansion rules. It presents comparative retention charts, tier allocation, an active memory table, live patient triage, and machine-learning validation metrics. The Flask API exposes the same classifier, triage, simulation, admitted-page, and capacity-evolution functionality programmatically.

---

# Important clarification for your report

The dashboard is a presentation layer. It does not independently calculate a different algorithm.

Its workflow is:

```text
Dashboard controls
    ↓
Parameter dictionary
    ↓
MimicLivingMemorySimulator.run_simulation()
    ↓
Scored and retained pages
    ↓
Charts, tables, cards, and downloads
```

The simulator is the core decision-making component. The dashboard only:

- Collects configuration.
- Calls the simulator.
- Displays returned results.
- Filters retained records.
- Provides triage inputs.
- Visualizes model metrics.

The main core algorithm is implemented in `mimic_simulator.py`, while the dashboard presentation is implemented in `dashboard_new.py`.