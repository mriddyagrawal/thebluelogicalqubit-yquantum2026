"""
Run MPO pipeline on a QASM file.

Usage:
    python run_peak.py circuit.qasm
    python run_peak.py circuit.qasm --max-bond 512 --threshold 128 --samples 5000
"""

import argparse
from qiskit import qasm2
from mpo_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Find peak bitstring of a QASM circuit")
    parser.add_argument("qasm_file", help="Path to .qasm file")
    parser.add_argument("--max-bond", type=int, default=256,
                        help="Max bond dimension for SVD truncation (default: 256)")
    parser.add_argument("--threshold", type=int, default=64,
                        help="Bond threshold that triggers unswapping (default: 64)")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of MPS samples (default: 5000)")
    parser.add_argument("--check-every", type=int, default=50,
                        help="Check threshold every N gates (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for transpilation (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    qc = qasm2.load(
        args.qasm_file,
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    qc = qc.remove_final_measurements(inplace=False)

    print(f"Circuit: {args.qasm_file}")
    print(f"Qubits: {qc.num_qubits}, Gates: {qc.size()}")
    print(f"Config: max_bond={args.max_bond}, threshold={args.threshold}, "
          f"samples={args.samples}")
    print()

    result = run_pipeline(
        qc,
        max_bond=args.max_bond,
        bond_threshold=args.threshold,
        check_every=args.check_every,
        n_samples=args.samples,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print(f"\n{'='*60}")
    print(f"  PEAK BITSTRING: {result['peak_bitstring_qiskit']}")
    print(f"{'='*60}")
    print(f"  Greedy:    {result['greedy_bitstring_qiskit']}")
    print(f"  Max bond:  {max(result['final_bonds'])}")
    print(f"  Unswaps:   {result['total_unswaps']}")
    print(f"  Time:      {result['total_time']:.1f}s")


if __name__ == "__main__":
    main()
