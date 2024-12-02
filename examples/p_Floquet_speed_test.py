import matplotlib.pyplot as plt
import mpl_setup
import numpy as np

import qutip as qt
import time

PI = np.pi


## ====================================================
## Defining the system and initial states
## ====================================================

# ----------------------------------------
# Defining the Hamiltonian
# ----------------------------------------
delta = 0.2 * 2*PI
epsilon = 1.0 * 2*PI
A = 2.5 * 2*PI
omega = 1.0 * 2*PI
T = 2*PI / omega

# Initial condition
psi0 = qt.basis(2, 0)

# Hamiltonian
H0 = -1 / 2 * (epsilon * qt.sigmaz() + delta * qt.sigmax())
H1 = A / 2 * qt.sigmax()
args = {'w': omega}
H = [H0, [H1, lambda t, w: np.sin(w * t)]]

# Numerical parameters
dt = .01
N_T = 10  # Number of periods
tlist = np.arange(0.0, N_T * T, dt)  # time-list


# ======================================================
# Initialization of arrays
# ======================================================
computational_time_floquet = np.zeros_like(tlist)
computational_time_sesolve = np.zeros_like(tlist)

expect_floquet = np.zeros_like(tlist)
expect_sesolve = np.zeros_like(tlist)


# ======================================================
# Starting simulation
# ======================================================
for n, t in enumerate(tlist):
    # --------------------------------
    # Floquet basis
    # --------------------------------
    tic_f = time.perf_counter()
    floquetbasis = qt.FloquetBasis(H, T, args)
    # Decomposing inital state into Floquet modes
    f_coeff = floquetbasis.to_floquet_basis(psi0)
    # Obtain evolved state in the original basis
    psi_t = floquetbasis.from_floquet_basis(f_coeff, t)
    p_ex = qt.expect(qt.sigmaz(), psi_t)
    toc_f = time.perf_counter()

    # Saving data
    computational_time_floquet[n] = toc_f - tic_f
    expect_floquet[n] = p_ex

    # --------------------------------
    # sesolve
    # --------------------------------
    tic_f = time.perf_counter()
    p_ex_ref = qt.sesolve(H, psi0, tlist[:n + 1], e_ops=[qt.sigmaz()], args=args).expect[0]
    toc_f = time.perf_counter()
    # Saving data
    computational_time_sesolve[n] = toc_f - tic_f
    expect_sesolve[n] = p_ex_ref[-1]


# ======================================================
# Plotting results
# ======================================================
sty = ["-", "--"]
fig, axs = plt.subplots(1, 2, figsize=(13.6, 4.5))
ax = axs[0]
ax.plot(tlist / T, computational_time_floquet, sty[0])
ax.plot(tlist / T, computational_time_sesolve, sty[1])

axs[0].set_yscale("log")
axs[0].set_yticks(np.logspace(-3, -1, 3))
axs[1].plot(tlist / T, np.real(expect_floquet), sty[0], lw='3.5')
axs[1].plot(tlist / T, np.real(expect_sesolve), sty[1], lw='3.5')

axs[0].set_xlabel(r'$t \, / \, T$')
axs[1].set_xlabel(r'$t \, / \, T$')
ax.set_ylabel(r'Computational Time [$s$]')
axs[1].set_ylabel(r'$\langle \sigma_z \rangle$')
ax.legend((r"Floquet", r"sesolve"), frameon=False)
xticks = np.rint(np.linspace(0, N_T, N_T + 1, endpoint=True))
axs[0].set_xticks(xticks)
axs[0].set_xlim([0, N_T])
axs[1].set_xticks(xticks)
axs[1].set_xlim([0, N_T])
axs[0].text(-.15, 1.05, r'(a)', transform=axs[0].transAxes)
axs[1].text(-.15, 1.05, r'(b)', transform=axs[1].transAxes)

fig.show()
fig.savefig('./floquet_speed_test.pdf')



exit()
# For formatting in paper, stripped leading whitespace

floquetbasis = qt.FloquetBasis(H, T, args)
# Decomposing inital state into Floquet modes
f_coeff = floquetbasis.to_floquet_basis(psi0)
# Obtain evolved state in the original basis
psi_t = floquetbasis.from_floquet_basis(f_coeff, t)