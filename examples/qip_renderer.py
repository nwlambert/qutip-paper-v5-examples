from qutip_qip.circuit import QubitCircuit

qc = QubitCircuit(2, num_cbits=1)
qc.add_gate("H", 0)
qc.add_gate("H", 1)
qc.add_gate("CNOT", 1, 0)
qc.add_measurement("M", targets=[0], classical_store=0)

# Visualize the circuit
qc.draw("latex")
qc.draw("text")
qc.draw("matplotlib")
