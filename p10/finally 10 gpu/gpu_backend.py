"""
GPU Backend
============
Provides:
  xp        — CuPy on GPU, NumPy on CPU (for tensor operations)
  np_cpu    — always NumPy (for RNG, string ops, Qiskit interop)
  to_numpy  — convert xp array to numpy
"""

import numpy as np_cpu

try:
    import cupy as xp
    GPU = True

    def to_numpy(arr):
        return xp.asnumpy(arr)

except ImportError:
    xp = np_cpu
    GPU = False

    def to_numpy(arr):
        return arr

if GPU:
    try:
        name = xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"[GPU] CuPy on {name}")
    except Exception:
        print("[GPU] CuPy available")
else:
    print("[CPU] CuPy not found, using NumPy")
