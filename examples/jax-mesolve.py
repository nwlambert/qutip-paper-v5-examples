import qutip as qt

import jax
import jax.numpy as jnp
import qutip_jax  # noqa: F401

print(qt.qeye(3, dtype='jax').dtype.__name__)
#?\pythonoutput?JaxArray

print(qt.qeye(3, dtype='jaxdia').dtype.__name__)
#?\pythonoutput?JaxDia

def Ising_solve(N, g0, J0, gamma, tlist, options, data_type='CSR'):
    # N  : number of spins
    # g0 : splitting
    # J0 : couplings
    with qt.CoreOptions(default_dtype=data_type):

        #Setup operators for individual qubits
        sx_list, sy_list, sz_list = [], [], []
        for i in range(N):
            op_list = [qt.qeye(2)] * N
            op_list[i] = qt.sigmax()
            sx_list.append(qt.tensor(op_list))
            op_list[i] = qt.sigmay()
            sy_list.append(qt.tensor(op_list))
            op_list[i] = qt.sigmaz()
            sz_list.append(qt.tensor(op_list))

        # Hamiltonian - Energy splitting terms
        H = 0.
        for i in range(N):
            H += g0 * sz_list[i]

        # Interaction terms
        for n in range(N - 1):
            H += -J0 * sx_list[n] * sx_list[n + 1]

        # Collapse operator acting locally on single spin
        c_ops = [gamma * sx_list[N-1]]

        # Initial state
        state_list = [qt.basis(2, 1)] * (N-1)
        state_list.append(qt.basis(2, 0))
        psi0 = qt.tensor(state_list)

        result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=sz_list, options=options)
        return result, result.expect[-1]


from diffrax import PIDController, Tsit5

with jax.default_device(jax.devices("cpu")[0]):
    # System parameters
    N = 4
    g0 = 1
    J0 = 1.4
    gamma = 0.1

    # Simulation parameters
    tlist = jnp.linspace(0, 5, 100)
    options = {
        "normalize_output": False,
        "store_states": True,
        "method": "diffrax",
        "stepsize_controller": PIDController(rtol=qt.settings.core['rtol'],
                                             atol=qt.settings.core['atol']),
        "solver": Tsit5()
    }

    result_ising, sz1 = Ising_solve(N, g0, J0, gamma, tlist, options, data_type='jaxdia')
    print(sz1)


# Use JAX as the backend
qutip_jax.set_as_default()


qutip_jax.set_as_default(revert = True)