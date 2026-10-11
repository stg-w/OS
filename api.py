"""LivingMemoryOS v4 REST API entrypoint.

Run with: python api.py
"""

from src.api.server_new import run_server

if __name__ == "__main__":
    print("==================================================")
    print(" LivingMemoryOS v4 REST API Server Starting")
    print(" Supporting the MIMIC-IV clinical cohort")
    print(" Listening on http://localhost:5000")
    print("==================================================")
    run_server(host="0.0.0.0", port=5000)
