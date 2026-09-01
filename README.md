# LivingMemoryOS-CAMR

**Clinical-Aware Memory Replacement (CAMR) for Resource-Constrained Healthcare Devices**

A modular Python project featuring calibrated ML mortality prediction, Clinical Memory Score (CMS) calculation, tiered min-heap replacement pools, a Streamlit web frontend, and a REST API.

---

## 📁 Project Structure (`src/` Architecture)

```
OS/
├── src/                                  # Source code package
│   ├── __init__.py
│   ├── config.py                         # Configurations, weights, and constants
│   ├── models/                           # Machine learning & clinical scoring
│   │   ├── __init__.py
│   │   ├── classifier.py                 # Calibrated Random Forest (Isotonic)
│   │   └── scoring.py                    # Clinical Memory Score (CMS) & care escalation
│   ├── memory/                           # Memory management & replacement algorithms
│   │   ├── __init__.py
│   │   ├── tier_pool.py                  # O(log n) min-heap tier pool (Emergency, High, Normal)
│   │   ├── fifo_pool.py                  # Baseline FIFO replacement queue
│   │   └── simulator.py                  # Stream simulation and benchmarking engine
│   ├── api/                              # Backend REST service
│   │   ├── __init__.py
│   │   └── server.py                     # Flask API routes & CORS handling
│   └── ui/                               # Web dashboard
│       ├── __init__.py
│       └── dashboard.py                  # Streamlit application UI & Plotly charts
│
├── main.py                               # CLI Entrypoint (run benchmark)
├── app.py                                # Streamlit Web UI Entrypoint
├── api.py                                # REST API Entrypoint
├── requirements.txt                      # Project dependencies
├── ICU_Patient_Monitoring_..._15000.csv  # Telemetry dataset
└── livingmemory_results_v2.csv           # Exported memory pages
```

---

## 🚀 How to Run

### 1. Run the Streamlit Web Application
To launch the interactive dashboard:
```bash
streamlit run app.py
```
> Opens in your browser at `http://localhost:8501`.

**Features in Streamlit UI:**
- **Simulation & Benchmark:** Live comparison of LivingMemoryOS vs. FIFO across memory pools, interactive capacity sliders, and Plotly charts.
- **Real-Time Patient Triage:** Sliders for 10 vital signs and biomarkers to compute instant mortality risk, care level (ER, ICU, HDU, WARD), CMS score, and assigned tier.
- **Model Performance:** Calibration curve analysis, ROC-AUC, PR-AUC, and Confusion Matrix.
- **Data Export:** Filter admitted memory pages and download CSV.

---

### 2. Run the CLI Benchmark
To train the model and execute the memory replacement simulation in terminal:
```bash
python main.py
```

---

### 3. Run the REST API Backend (Optional)
To start the localhost REST server on port 5000:
```bash
python api.py
```

#### Available Endpoints:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Model status, metrics, and default configs |
| `POST` | `/api/predict` | Single patient triage evaluation |
| `POST` | `/api/simulate` | Run memory replacement simulation |

---

## ⚙️ Core Concepts

1. **Calibrated ML Probabilities**: Isotonic calibration prevents uncalibrated Random Forest overconfidence, yielding true clinical risk.
2. **Tiered Min-Heap Budgeting**: Separate bounded min-heaps (`EMERGENCY`, `HIGH_PRIORITY`, `NORMAL`) replace victims in $O(\log n)$ time.
3. **Non-Evictable Emergency Tier**: Prevents high-risk critical ICU patients from ever being displaced by normal telemetry pages.
