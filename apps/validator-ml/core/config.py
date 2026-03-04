import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, "..", "..", ".."))  # -> limitwear-validator
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
