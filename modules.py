from qiskit import QuantumCircuit, transpile
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise
from pathlib import Path

def load_qasm(qasm_path: str) -> QuantumCircuit:
    """Load an OpenQASM file and return it as a Qiskit circuit.

    Args:
        qasm_path: Path to the OpenQASM file.

    Returns:
        QuantumCircuit parsed from the file contents.
    """
    with open(qasm_path, "r") as f:
        qasm = f.read()
    return QuantumCircuit.from_qasm_str(qasm)

def remove_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of a circuit with all barrier operations removed.

    Args:
        qc: Input circuit that may contain barriers.

    Returns:
        A new QuantumCircuit without barrier instructions.
    """
    qc_no_barriers = qc.copy()
    qc_no_barriers.data = [inst for inst in qc_no_barriers.data if inst.operation.name != 'barrier']
    return qc_no_barriers

def qiskit_optimize_no_coupling(qc: QuantumCircuit, optimization_level: int = 3) -> QuantumCircuit:
    """Optimize a circuit with Qiskit assuming no hardware coupling constraints.

    The transpiler uses level-3 optimization and keeps the circuit within the
    gate set already present in the input circuit.

    Args:
        qc: Circuit to optimize.
        optimization_level: Level of optimization to apply.

    Returns:
        The optimized circuit produced by Qiskit's transpiler.
    """
    return transpile(
        qc,
        basis_gates= list(qc.count_ops().keys()),
        optimization_level=3,
        coupling_map=None
    )

def tket_optimize(qc: QuantumCircuit, swap_later: bool = True) -> QuantumCircuit:
    """Optimize a circuit using pytket's FullPeepholeOptimise pass.

    Args:
        qc: Input Qiskit circuit.
        swap_later: Whether to replace implicit swaps when converting back to
            Qiskit. Set to False to keep swaps implicit and handle remapping
            later.

    Returns:
        Optimized circuit converted back to QuantumCircuit.
    """
    tk_circ = qiskit_to_tk(qc)
    FullPeepholeOptimise().apply(tk_circ)
    return tk_to_qiskit(tk_circ, replace_implicit_swaps=swap_later)

def qc_optimize(qc: QuantumCircuit, first: str = "tket", swap_later: bool = True) -> QuantumCircuit:
    """Run a two-stage optimization pipeline combining pytket and Qiskit.

    Args:
        qc: Circuit to optimize.
        first: Optimizer to run first. Must be either "tket" or "qiskit".
        swap_later: Forwarded to `tket_optimize` to control implicit swap
            replacement.

    Returns:
        Circuit optimized by both toolchains in the selected order.

    Raises:
        ValueError: If `first` is not "tket" or "qiskit".
    """
    if first == "tket":
        temp_qc = tket_optimize(qc, swap_later=swap_later)
        return qiskit_optimize_no_coupling(temp_qc)
    elif first == "qiskit":
        temp_qc = qiskit_optimize_no_coupling(qc)
        return tket_optimize(temp_qc, swap_later=swap_later)
    else:
        raise ValueError("first must be 'tket' or 'qiskit'")

def run_all_optimizations(qc: QuantumCircuit) -> dict[str, QuantumCircuit]:
    """Run all optimization variants and return them in a dictionary.

    This runs both possible orders of the two-stage optimization pipeline, and
    for the tket stage it runs both with and without implicit swap replacement.

    Args:
        qc: Circuit to optimize.

    Returns:
        A dictionary mapping optimization variant names to their results.

    Notes:
        Prints a concise summary (gate count, depth, and operation breakdown)
        for the input circuit and each optimization variant before returning.
    """

    results = {}

    processes = ["tket_first", "qiskit_first", "tket_first_no_swaps", "qiskit_first_no_swaps"]

    print(f"Original: gates={qc.size()}, depth={qc.depth()}")
    print(f"Gates: {qc.count_ops()}")
    print()

    for process in processes:
        print(f"Now running {process} optimization...")
        results[process] = qc_optimize(qc, first=process.split("_")[0], swap_later=not process.endswith("no_swaps"))
        optimized_qc = results[process]
        print(f"{process}: gates={optimized_qc.size()}, depth={optimized_qc.depth()}")
        print(f"Gates: {optimized_qc.count_ops()}")
        print()

    return results

qasm_paths = {
    1: "P1_little_peak.qasm",
    2: "P2_small_bump.qasm",
    3: "P3_tiny_ripple.qasm",
    4: "P4_gentle_mound.qasm",
    5: "P5_soft_rise.qasm",
    6: "P6_low_hill.qasm",
    7: "P7_rolling_ridge.qasm",
    8: "P8_bold_peak.qasm",
    9: "P9_grand_summit.qasm",
    10: "P10_eternal_mountain.qasm",
}

repo_root = Path("/Users/mridul/Desktop/Projects/thelogicalqubit-yquantum2026/")

def get_qc(problem_number: int, qasm_name: str | None = None) -> QuantumCircuit:
    qasm_name = qasm_name or qasm_paths[problem_number]
    qasm_path = repo_root / f"P{problem_number}" / qasm_name
    if not qasm_path.exists():
        raise FileNotFoundError(f"Could not find {qasm_path}")

    return load_qasm(str(qasm_path))

def draw_problem(problem_number: int) -> None:
    qc = get_qc(problem_number)
    display(qc.draw("mpl", fold=-1))