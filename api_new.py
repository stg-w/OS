"""
LivingMemoryOS - Next-Gen REST API Server (MIMIC-IV Clinical Cohort)
Dedicated backend service for LivingMemoryOS v4.
Run with: python api_new.py
"""

from src.api.server import app
from src.api.server_new import run_server

if __name__ == "__main__":
    print("==========================================================")
    print(" LivingMemoryOS v4 - MIMIC-IV Clinical REST API Service")
    print(" 550,818 Inpatient Admissions & ICU Stays Cohort Engine")
    print(" Listening on http://localhost:5000")
    print(" Available v2 endpoints:")
    print("   GET  /api/v2/status")
    print("   POST /api/v2/predict")
    print("   POST /api/v2/simulate")
    print("   GET  /api/v2/admitted")
    print("   GET  /api/v2/capacity-evolution")
    print(" Available endpoints:")
    print("   GET  /api/status")
    print("   POST /api/predict")
    print("   POST /api/simulate")
    print("   GET  /api/admitted")
    print("   GET  /api/capacity-evolution")
    print("==========================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
    run_server(host="0.0.0.0", port=5000)

