#!/bin/bash
#PBS -N mpo_impl_gpu
#PBS -A CESM0029
#PBS -q main
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=32:ngpus=1:mem=80GB
#PBS -o mpo_impl_gpu.out
#PBS -e mpo_impl_gpu.err
#PBS -j oe
#PBS -V

set -euo pipefail

module load cuda

# cd to repo root and sync deps.
cd /glade/u/home/sah/mridul/thelogicalqubit-yquantum2026
uv sync

cd "Finally 10 GPU"
uv run python -u mpo_implementation.py P10_eternal_mountain.qasm --max-bond 256 --threshold 9999 --check-every 500 --samples 2000 --seed 42