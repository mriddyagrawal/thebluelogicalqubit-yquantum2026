# YQuantum 2026 - BlueQubit Challenge
This document summarizes the challenges and our approach to each problem of the BlueQubit Challenge, as part of YQuantum 2026

## Problems and solutions

### Problem 1: Little Dimple 🫧
**qubits: 4**  
**Points: 10**  
Solved ✅

"This circuit is intentionally simple — inspect the gates and their order carefully. The peak bitstring can be inferred directly from the visual structure without running any simulator."

Our approach: We extracted the bitstring by retreiving the exact statevector.  
Final bitstring: `1001`


### Problem 2: Small Bump 🪨
**qubits: 12**  
**Points: 20**  
Solved ✅

"A full classical statevector simulation on a CPU is sufficient here. Try running the circuit end-to-end and inspect the output amplitudes to identify the dominant bitstring. This can be solved on BlueQubit’s quantum device as well !"

Our approach: We extracted the bitstring by retreiving the exact statevector.  
Final bitstring:


### Problem 3: Tiny Ripple 🌊
**qubits: 30**  
**Points: 30**  
Solved ✅

"Similar to the previous challenge, brute-force classical simulation is still viable. Focus on extracting the highest-probability outcome from the final state rather than optimizing compilation."

Our approach:   
Final bitstring:

### Problem 4: Gentle Mound 🌿
**qubits: 40**  
**Points: 40**  
Solved ✅

"The circuit size starts pushing statevector limits, but its depth and entanglement remain manageable. Approximate simulators like MPS can be very handy!"

Our approach:   
Final bitstring:

### Problem 5: Soft Rise 🌄
**qubits: 50**  
**Points: 50**  
Solved ✅

"If your simulator returns a near-uniform  flat distribution, the circuit is likely highly entangling or scrambling. Increase bond dimension, sample multiple runs and use the fact that the high signal of the peak bitstring can be “hidden” in the flat distribution."

Our approach:   
Final bitstring: `00011011001101000001010110110100101010011000011001`

### Problem 6: Low Hill ⛰️
**qubits: 60**  
**Points: 60**  
Solved ✅

"Direct simulation becomes impractical here. Instead, search for canceling gates and simplifying the circuit before simulating it. Qiskit has good tools for approximate transpilation and simplification of circuits."

Our approach:   
Final bitstring: `101100101001010001110111100101100011101011100000000000110111`

### Problem 7: Rolling Ridge 🏞️
**qubits: 42**  
**Points: 70**  
Solved ✅

"Exact or approximate simulations might not work here. Instead try to analyze the circuit and spot any simplifications you can make!"

Our approach:   
Final bitstring:

### Problem 8: Bold Peak 🏜
**qubits: 58**  
**Points: 80**  
Solved ✅

"This challenge blends multiple patterns from earlier problems. Combine visual inspection, partial simulation, decomposition, and compilation tricks to expose the hidden bias toward the peak bitstring."

Our approach:   
Final bitstring:

### Problem 9: Grand Summit 🏔️
**qubits: 69**  
**Points: 90**  
Solved ✅

"This challenge is similar to the previous one, but is much bigger."

Our approach:   
Final bitstring: `010001110101001111111110011111111111000010000000111101010111011100011`

### Problem 10: Eternal Mountain 🗻
**qubits: 56**  
**Points: 100**  
Solved ⚠️

"Use everything you learned in previous problems and try to solve the final circuit!"

Our approach:   
Final bitstring:
