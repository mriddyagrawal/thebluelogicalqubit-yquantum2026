from qiskit import QuantumCircuit

qc = QuantumCircuit.from_qasm_file('P1_little_peak.qasm')

print(qc.draw())