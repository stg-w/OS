"""
LivingMemoryOS - REST API Entrypoint
Run with: python api.py
"""

from src.api.server import run_server

if __name__ == "__main__":
    print("==================================================")
    print(" LivingMemoryOS REST API Server Starting")
    print(" Listening on http://localhost:5000")
    print("==================================================")
    run_server(host="0.0.0.0", port=5000)
