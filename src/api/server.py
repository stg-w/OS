"""
Flask REST API Server Module
Provides HTTP API endpoints for LivingMemoryOS simulation and triage.
"""

from flask import Flask, request, jsonify
from src.config import (
    DEFAULT_TIER_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_CRITICAL_THRESHOLD,
    FEATURES,
)
from src.models.classifier import MortalityClassifier
from src.memory.simulator import LivingMemorySimulator

app = Flask(__name__)

_simulator_instance = None


def get_simulator():
    global _simulator_instance
    if _simulator_instance is None:
        classifier = MortalityClassifier()
        classifier.load_and_train()
        _simulator_instance = LivingMemorySimulator(classifier)
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
        "features": FEATURES,
        "default_capacities": DEFAULT_TIER_CAPACITY,
        "default_weights": DEFAULT_CMS_WEIGHTS,
        "default_critical_threshold": DEFAULT_CRITICAL_THRESHOLD,
        "metrics": sim.classifier.metrics,
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_patient():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}
    vitals = {f: float(data.get(f, 0.0)) for f in FEATURES}
    patient_id = str(data.get("patient_id", "PT_001"))
    custom_weights = data.get("weights", DEFAULT_CMS_WEIGHTS)
    threshold = float(data.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD))

    sim = get_simulator()
    result = sim.triage_single_patient(
        vitals=vitals,
        patient_id=patient_id,
        cms_weights=custom_weights,
        critical_threshold=threshold,
    )
    return jsonify({"success": True, "result": result})


@app.route("/api/simulate", methods=["POST", "OPTIONS"])
def run_simulation_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}
    capacities = data.get("tier_capacity", DEFAULT_TIER_CAPACITY)
    weights = data.get("weights", DEFAULT_CMS_WEIGHTS)
    threshold = float(data.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD))

    sim = get_simulator()
    sim_results = sim.run_simulation(
        tier_capacities=capacities,
        cms_weights=weights,
        critical_threshold=threshold,
    )

    return jsonify({
        "success": True,
        "total_stream_size": sim_results["total_test_stream_size"],
        "fifo_size": sim_results["fifo_size"],
        "fifo_critical_pages": sim_results["fifo_critical_pages"],
        "living_critical_pages": sim_results["living_critical_pages"],
        "improvement_pct": sim_results["improvement_pct"],
        "rejected_emergency": sim_results["rejected_emergency"],
        "tier_summary": sim_results["tier_summary"],
        "living_memory": sim_results["living_memory"],
    })


def run_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False)
