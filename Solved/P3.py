import time
import warnings
import sys
import os
from qiskit import QuantumCircuit
import numpy as np
from qiskit import QuantumCircuit, transpile
import requests
import bluequbit
from pathlib import Path
sys.path.append(os.path.abspath("../.."))
from modules import *

# os.environ["BLUEQUBIT_PPS_DO_NO_USE_PARALLEL_COMPUTE"] = "1"
os.environ["BLUEQUBIT_DEQART_INTERNAL_DISABLE_STRICT_VALIDATIONS"] = "1"


def load_simple_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Notebook is under tutorial/, so check both cwd and its parent for .env
load_simple_env(Path.cwd() / ".env")
load_simple_env(Path.cwd().parent / ".env")
load_simple_env(Path.cwd().parent.parent / ".env")

api_token = os.getenv("bluequbitapi") or os.getenv("BLUEQUBIT_API_TOKEN")
if not api_token:
    raise ValueError("Missing BlueQubit API token. Set bluequbitapi in .env")

# BlueQubit SDK reads BLUEQUBIT_API_TOKEN by default.
os.environ["BLUEQUBIT_API_TOKEN"] = api_token

warnings.filterwarnings("ignore", category=UserWarning)

bq = bluequbit.init(api_token=api_token)

qc = load_qasm('../P3_tiny_ripple.qasm')

print(f"Gates: {qc.count_ops()}")
qc.draw("mpl", fold=-1)

results = bq.run(qc)
probs = results.get_counts()

top10_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]) # get top 10
print(top10_probs) 