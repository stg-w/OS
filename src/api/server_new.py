"""
Flask REST API Server Module for LivingMemoryOS v4 (MIMIC-IV Clinical Cohort)
Dedicated backend service providing endpoints for clinical risk triage and memory simulation.
"""

import os
from flask import Flask, request, jsonify
import pandas as pd
from src.config_new import (
    FEATURES,
    DEFAULT_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_PROTECTED_THRESHOLDS,
    EMERGENCY_CMS_THRESHOLD,
    RESULTS_V4_PATH,
    CAPACITY_V4_PATH,
)
from src.models.mimic_classifier import MimicMortalityClassifier
from src.memory.mimic_simulator import MimicLivingMemorySimulator

app = Flask(__name__)

_simulator_instance = None


def get_simulator():
    global _simulator_instance
    if _simulator_instance is None:
        classifier = MimicMortalityClassifier()
        classifier.load_and_train()
        _simulator_instance = MimicLivingMemorySimulator(classifier)
    return _simulator_instance


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


@app.route("/api/status", methods=["GET"])
def get_status():
    sim = get_simulator()
    return jsonify({
        "status": "ready",
        "dataset": "MIMIC-IV Real Clinical Cohort (550,818 admissions)",
        "features": FEATURES,
        "default_capacity": DEFAULT_CAPACITY,
        "default_weights": DEFAULT_CMS_WEIGHTS,
        "default_protected_thresholds": DEFAULT_PROTECTED_THRESHOLDS,
        "emergency_cms_threshold": EMERGENCY_CMS_THRESHOLD,
        "metrics": sim.classifier.metrics,
        "cohort_summary": sim.classifier.cohort_summary,
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_patient():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}
    sim = get_simulator()

    anchor_age = float(data.get("anchor_age", data.get("age_risk", 0.5) * 100.0))
    los = float(data.get("los", data.get("icu_risk", 0.1) * 10.0))
    transfer_count = float(data.get("transfer_count", data.get("escalation_risk", 0.3) * 10.0))
    abnormal = float(data.get("abnormal", data.get("biomarker_risk", 0.4) * 20.0))
    admission_type = str(data.get("admission_type", "EMERGENCY"))
    first_careunit = str(data.get("first_careunit", "MICU"))
    subject_id = str(data.get("subject_id", "LIVE_PT_001"))

    custom_weights = data.get("weights", DEFAULT_CMS_WEIGHTS)
    protected_thresholds = data.get("protected_thresholds", DEFAULT_PROTECTED_THRESHOLDS)

    result = sim.triage_single_patient(
        anchor_age=anchor_age,
        los=los,
        transfer_count=transfer_count,
        abnormal=abnormal,
        admission_type=admission_type,
        first_careunit=first_careunit,
        subject_id=subject_id,
        cms_weights=custom_weights,
        protected_thresholds=protected_thresholds,
    )
    return jsonify({"success": True, "result": result})


@app.route("/api/simulate", methods=["POST", "OPTIONS"])
def run_simulation_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}
    memory_size = int(data.get("memory_size", DEFAULT_CAPACITY))
    stream_size = int(data.get("stream_size", 5000))
    weights = data.get("weights", DEFAULT_CMS_WEIGHTS)
    protected_thresholds = data.get("protected_thresholds", DEFAULT_PROTECTED_THRESHOLDS)
    emergency_threshold = float(data.get("emergency_threshold", EMERGENCY_CMS_THRESHOLD))

    sim = get_simulator()
    sim_results = sim.run_simulation(
        memory_size=memory_size,
        stream_size=stream_size,
        cms_weights=weights,
        protected_thresholds=protected_thresholds,
        emergency_threshold=emergency_threshold,
    )

    return jsonify({
        "success": True,
        "stream_size": sim_results["stream_size"],
        "memory_size": sim_results["memory_size"],
        "adaptive_capacity": sim_results["adaptive_capacity"],
        "fifo_average_cms": sim_results["fifo_average_cms"],
        "living_average_cms": sim_results["living_average_cms"],
        "improvement_pct": sim_results["improvement_pct"],
        "protected_pages": sim_results["protected_pages"],
        "fifo_protected_pages": sim_results["fifo_protected_pages"],
        "high_priority_pages": sim_results["high_priority_pages"],
        "emergency_pages": sim_results["emergency_pages"],
        "rejected_count": sim_results["rejected_count"],
        "eviction_count": sim_results["eviction_count"],
        "living_memory": sim_results["living_memory"],
    })


@app.route("/api/admitted", methods=["GET"])
def get_admitted():
    if os.path.exists(RESULTS_V4_PATH):
        df = pd.read_csv(RESULTS_V4_PATH)
        return jsonify({
            "success": True,
            "count": len(df),
            "pages": df.to_dict(orient="records"),
        })
    sim = get_simulator()
    res = sim.run_simulation(stream_size=1000)
    return jsonify({
        "success": True,
        "count": len(res["living_memory"]),
        "pages": res["living_memory"],
    })


@app.route("/api/capacity-evolution", methods=["GET"])
def get_capacity_evolution():
    if os.path.exists(CAPACITY_V4_PATH):
        df = pd.read_csv(CAPACITY_V4_PATH)
        return jsonify({
            "success": True,
            "data": df.to_dict(orient="records"),
        })
    return jsonify({
        "success": True,
        "data": [{
            "initial_capacity": 100,
            "adaptive_capacity": 100,
            "emergency_pages": 0,
            "protected_pages": 100,
        }],
    })


def run_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False)

