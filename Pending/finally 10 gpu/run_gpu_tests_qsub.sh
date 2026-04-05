#!/bin/bash
#PBS -N mpo_gpu_tests
#PBS -A CESM0029
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:ngpus=1:mem=64GB
#PBS -o mpo_gpu_tests.out
#PBS -e mpo_gpu_tests.err
#PBS -j oe
#PBS -V

set -euo pipefail

module load cuda

# cd to repo root and sync deps.
cd /glade/u/home/sah/mridul/thelogicalqubit-yquantum2026
uv sync

cd "Finally 10 GPU"

echo "=========================================="
echo "  Running mpo_tests.py (GPU)"
echo "=========================================="
uv run python -u mpo_tests.py

echo ""
echo "=========================================="
echo "  Running mpo_pipeline.py (GPU)"
echo "=========================================="
uv run python -u mpo_pipeline.py

echo ""
echo "=========================================="
echo "  Running mpo_routing_tests.py (GPU)"
echo "=========================================="
uv run python -u mpo_routing_tests.py

echo ""
echo "=========================================="
echo "  ALL GPU TESTS COMPLETE"
echo "=========================================="