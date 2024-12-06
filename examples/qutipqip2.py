#%%
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

from qutip import basis, tensor
import qutip as qt
qt.settings.core["default_dtype"] = "csr"
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits


# Parameters
epsilon1 = 0.7
epsilon2 = 1.
g = 0.3
tf = 20.  # total time
dt = 2.  # Trotter step size
num_steps = int(tf/dt)

# Hamiltonian we are modelling
sx1 = qt.sigmax() & qt.qeye(2)
sx2 = qt.qeye(2) & qt.sigmax()

sy1 = qt.sigmay() & qt.qeye(2)
sy2 = qt.qeye(2) & qt.sigmay()

sz1 = qt.sigmaz() & qt.qeye(2)
sz2 = qt.qeye(2) & qt.sigmaz()

H = 0.5 * epsilon1 * sz1 + 0.5 * epsilon2 * sz2 + g * sx1 * sx2

init_state = basis(2, 0) & basis(2, 1)


# Exact Schrodinger equation

states_sesolve = qt.sesolve(H, init_state, tlist=np.linspace(0, tf, 100)).states


# Trotterization

trotter_simulation = QubitCircuit(2)

trotter_simulation.add_gate("RZ", targets=[0], arg_value=(epsilon1 * dt))
trotter_simulation.add_gate("RZ", targets=[1], arg_value=(epsilon2 * dt))

trotter_simulation.add_gate("H", targets=[0])
trotter_simulation.add_gate("RZX", targets=[0, 1], arg_value=g * dt * 2)
trotter_simulation.add_gate("H", targets=[0])

trotter_simulation.compute_unitary()
plot1 = trotter_simulation.draw(renderer="matplotlib")
# Evaluate multiple iteration of a circuit
result_circ = init_state
state_trotter_circ = [init_state]

for dd in range(num_steps):
    result_circ = trotter_simulation.run(state=result_circ)
    state_trotter_circ.append(result_circ)

# Run pulse-level noisy hardware simulation

processor = SCQubits(num_qubits=2, t1=2e5, t2=2e5)
processor.load_circuit(trotter_simulation)
# Since SCQubit are modelled as qutrit, we need three-level systems here.
init_state = tensor(basis(3, 0), basis(3, 1))

state_proc = init_state
state_list_proc = [init_state]
for dd in range(num_steps):
    result = processor.run_state(state_proc)
    state_proc = result.final_state
    state_list_proc.append(result.final_state)


# Plotting

fig, ax = processor.plot_pulses(figsize=(8, 5))
plt.savefig("circuit1.pdf")
fig.show()

plt.figure(figsize=(10, 4.54))
sigmaz_qutrit = qt.basis(3, 0)*qt.basis(3, 0).dag() - qt.basis(3, 1)*qt.basis(3, 1).dag()
plt.plot(np.linspace(0, tf, 100), qt.expect(qt.sigmaz() & qt.qeye(2),states_sesolve), '-',label='Ideal')
plt.plot(np.linspace(0, tf, num_steps + 1), qt.expect(qt.sigmaz() & qt.qeye(2), state_trotter_circ), '--d', label=r'Trotter circuit')
plt.plot(np.linspace(0, tf, num_steps + 1), qt.expect(sigmaz_qutrit & qt.qeye(3), state_list_proc), '-.o',label='noisy hardware')

plt.xlabel('Time', fontsize=18)
plt.ylabel(r'$\langle \sigma_z^{(1)} \rangle$', fontsize=18)

plt.legend(loc=0, fontsize=18, bbox_to_anchor=(1.1, 1.05))
plt.savefig("qip1.pdf")
plt.show()

#%%
# Lindblad simulation
epsilon1 = 0.7
epsilon2 = 1.
g = 0.3
gam = 0.03 # dissipation rate
tf = 20.  # total time
dt = 2.  # Trotter step size
num_steps = int(tf/dt)

# Normal Master equation simulation
init_state_system = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
tlist = np.linspace(0, tf, 100)
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
sx2 = qt.tensor(qt.qeye(2), qt.sigmax())

H = 0.5 * epsilon1 * sz1 + 0.5 * epsilon2 * sz2 + g * sx1 * sx2

# Collapse operators
sm1 = qt.tensor(qt.destroy(2).dag(), qt.qeye(2))
sm2 = qt.tensor(qt.qeye(2), qt.destroy(2).dag())
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]

result_me = qt.mesolve(
    H, init_state_system, tlist, c_ops,
    e_ops = [tensor([qt.sigmaz(), qt.qeye(2)]), sz2]
    )

# Trotterized circuit simulation
trotter_simulation_noisey = QubitCircuit(4)

# Coherent dynamics
trotter_simulation_noisey.add_gate("RZ",targets=[1], arg_value=epsilon1*dt)
trotter_simulation_noisey.add_gate("RZ",targets=[2], arg_value=epsilon2*dt)

trotter_simulation_noisey.add_gate("H", targets=[1])
trotter_simulation_noisey.add_gate("RZX", targets=[1, 2], arg_value=g*dt*2)
trotter_simulation_noisey.add_gate("H", targets=[1])

# Decoherence
# exp(-i XX t)
trotter_simulation_noisey.add_gate("H", targets=[0])
trotter_simulation_noisey.add_gate("RZX", targets=[0, 1], arg_value=sqrt(gam)*sqrt(dt))
trotter_simulation_noisey.add_gate("H", targets=[0])

# exp(-i YY t)
trotter_simulation_noisey.add_gate("RZ", 1, arg_value=np.pi/2)
trotter_simulation_noisey.add_gate("RX", 0, arg_value=-np.pi/2)
trotter_simulation_noisey.add_gate("RZX", [0, 1], arg_value=sqrt(gam)*sqrt(dt))
trotter_simulation_noisey.add_gate("RZ", 1, arg_value=-np.pi/2)
trotter_simulation_noisey.add_gate("RX", 0, arg_value=np.pi/2)

# exp(-i XX t)
trotter_simulation_noisey.add_gate("H", targets=[2])
trotter_simulation_noisey.add_gate("RZX", targets=[2, 3], arg_value=sqrt(gam)*sqrt(dt))
trotter_simulation_noisey.add_gate("H", targets=[2])

# exp(-i YY t)
trotter_simulation_noisey.add_gate("RZ", 3, arg_value=np.pi/2)
trotter_simulation_noisey.add_gate("RX", 2, arg_value=-np.pi/2)
trotter_simulation_noisey.add_gate("RZX", [2, 3], arg_value=sqrt(gam)*sqrt(dt))
trotter_simulation_noisey.add_gate("RZ", 3, arg_value=-np.pi/2)
trotter_simulation_noisey.add_gate("RX", 2, arg_value=np.pi/2)
trotter_simulation_noisey.draw(renderer="matplotlib")

state_system = qt.ket2dm(init_state_system)
state_trotter_circ = [init_state_system]
for dd in range(num_steps):
    state_full = tensor(
        basis(2, 1)*basis(2, 1).dag(),
        state_system,
        basis(2, 1)*basis(2, 1).dag()
        )
    state_full = trotter_simulation_noisey.run(state=state_full)
    state_system = state_full.ptrace([1, 2])
    state_trotter_circ.append(state_system)
    

# plot results 
times_circ = np.arange(0, tf + dt, dt)
plt.figure()
plt.plot(tlist, result_me.expect[0], '-', label=r'Ideal')
plt.plot(times_circ, qt.expect(qt.tensor(qt.sigmaz(), qt.qeye(2)), state_trotter_circ), "--d", lw=2, label=r'trotter')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Expectation values', fontsize=18)
plt.legend()
plt.show()


#%% Pulse-level noisy hardware simulation, takes ~20 min
processor = SCQubits(num_qubits=4, t1=3.e4, t2=3.e4)
processor.load_circuit(trotter_simulation_noisey)
init_state_system = tensor(basis(3, 0), basis(3, 1))
fig, ax = processor.plot_pulses(figsize=(8, 5))
plt.savefig("circuit2.pdf")
fig.show()

state_system = qt.ket2dm(init_state_system)
state_list_proc = [state_system]
for dd in range(num_steps):
    state_full = tensor(
            basis(3, 1) * basis(3, 1).dag(),
            state_system,
            basis(3, 1) * basis(3, 1).dag(),
            )
    result_noisey = processor.run_state(
        state_full,
        solver="mesolve",
        options={
            "store_states": False,
            "store_final_state": True,
        }
        )
    state_full = result_noisey.final_state
    state_system = state_full.ptrace([1,2])
    state_list_proc.append(state_system)
    print(f"Step {dd+1}/{num_steps} finished.")


times_circ = np.arange(0, tf + dt, dt)
#%%
plt.figure()
plt.plot(tlist, result_me.expect[0],'-', label=r'Ideal')
plt.plot(times_circ, qt.expect(qt.sigmaz() & qt.qeye(2), state_trotter_circ),
         "--d", label=r'Trotter circuit')
plt.plot(times_circ,
         qt.expect(sigmaz_qutrit & qt.qeye(3), state_list_proc),
         "-.o",
         label=r'noisy hardware')

plt.xlabel('Time$', fontsize=18)
plt.ylabel(r'$\langle \sigma_z^{(1)} \rangle$', fontsize=18)
plt.legend()

plt.savefig("qip2.pdf")
plt.show()

