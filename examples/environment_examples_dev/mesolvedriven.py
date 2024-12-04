import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, BosonicBath
from qutip import UnderDampedEnvironment, QobjEvo
import matplotlib.pyplot as plt
from mpl_setup import *


############ Normal driven qubit, time-dependent example 1 for paper ##########


# Hamiltonian parameters
Delta = 2 * np.pi  # qubit splitting
omega_d = Delta  # drive frequency
A = 0.01 * Delta  # drive amplitude

# Driving field


def f(t):
    return np.sin(omega_d * t)


# Time-dependent Hamiltonian
H0 = Delta / 2.0 * qt.sigmaz()
H1 = [A / 2.0 * qt.sigmax(), f]
H = [H0, H1]

# Bath parameters
gamma = 0.005 * Delta / (2 * np.pi)  # dissipation strength
temp = 0  # temperature

# Simulation parameters
psi0 = qt.basis(2, 0)  # initial state
e_ops = [qt.sigmaz()]

T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 1000 * T, 500)


# --- mesolve ---

c_ops_me = [np.sqrt(gamma) * qt.sigmam()]
me_result = qt.mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)


# --- mesolve, RWA ---

c_ops_me_RWA = [np.sqrt(gamma) * qt.sigmam()]
H_RWA = (Delta - omega_d) * 0.5 * qt.sigmaz() + A / 4 * qt.sigmax()
me_result_RWA = qt.mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)


# --- HEOM ---

# Numerical parameters
max_depth = 4  # number of hierarchy levels
fit_times = np.linspace(0, 20, 1000)  # range for correlation function fit

# Emulate flat spectral density with an underdamped spectral density
wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi
gamma_heom = 1.9 * w0
Gamma = gamma_heom / 2

Omega = np.sqrt(w0**2 - Gamma**2)
lambd = np.sqrt(
    0.5 * gamma / (gamma_heom * wsamp) *
    ((w0 ** 2 - wsamp ** 2) ** 2 + (gamma_heom ** 2) * ((wsamp) ** 2))
)
# Create Environment
bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=0)

# To handle zero-temperature underdamped bath, fit its correlation function
exp_bath, fit_info = bath.approx_by_cf_fit(fit_times, Ni_max=1, Nr_max=2, target_rsme=None)
print(fit_info['summary'])



# # Do simulation
HEOM_corr_fit = HEOMSolver(QobjEvo(H), (exp_bath, qt.sigmax()), max_depth=max_depth,
                    options={'nsteps': 15000, 'rtol': 1e-12, 'atol': 1e-12})
results_corr_fit = HEOM_corr_fit.run(psi0 * psi0.dag(), tlist, e_ops=e_ops)

# --- brmesolve ---

# Bose einstein distribution
def nth(w):
    if temp > 0:
        return 1 / (np.exp(w / temp) - 1)
    else:
        return 0

# Power spectrum
def power_spectrum(w):
    if w > 0:
        return gamma * (nth(w) + 1)
    elif w == 0:
        return 0
    else:
        return gamma * nth(-w)

a_ops = [[qt.sigmax(), power_spectrum]]
brme_result = qt.brmesolve(H, psi0, tlist, a_ops=a_ops,e_ops=e_ops, sec_cutoff=-1)


# --- Plots ---

plt.figure()

plt.plot(tlist, me_result.expect[0], '-', label=r'mesolve (time-dep)')
plt.plot(tlist, me_result_RWA.expect[0], '-.', label=r'mesolve (rwa)')
plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
plt.plot(tlist, brme_result.expect[0], ':', linewidth=6, label=r'brmesolve')

plt.xlabel(r'$t\, /\, \Delta^{-1}$', fontsize=18)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=18)
plt.legend()
plt.text(200, 0.7, "(a)", fontsize=18)

plt.savefig("mesolve_driven_1.pdf")


############ Frequency modulation, Example 2 ############

# hamiltonian parameters
omega_d = 0.05 * Delta  # drive frequency
A = Delta  # drive amplitude

# driving field


def f(t):
    return np.sin(omega_d * t)


H0 = [A / 2.0 * qt.sigmaz(), f]
H = [H0]

# bath parameters
gamma = 0.05 * Delta / (2 * np.pi)  # dissipation strength

# simulation parameters
psi0 = qt.basis(2, 0)  # initial state
e_ops = [qt.sigmaz()]

T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 2 * T, 400)

# --- HEOM ---

wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi

gamma_heom = 1.9 * w0
# gamma = 1.5 * w0


lambd = np.sqrt(
    0.5 * gamma *
    ((w0 ** 2 - wsamp ** 2) ** 2 + (gamma_heom ** 2) * ((wsamp) ** 2)) /
    (gamma_heom * wsamp))


# --- mesolve ---

c_ops_me = [np.sqrt(gamma) * qt.sigmam()]
me_result = qt.mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)


# --- mesolve, RWA ---

c_ops_me_RWA = [np.sqrt(gamma) * qt.sigmam()]
H_RWA = (Delta - omega_d) * 0.5 * qt.sigmaz() + A / 4 * qt.sigmax()
me_result_RWA = qt.mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)


# --- HEOM ---

# Create Environment
bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=0)
fit_times = np.linspace(0, 5, 1000)  # range for correlation function fit

# To handle zero-temperature underdamped, fit its correlation function

exp_bath, fit_info = bath.approx_by_cf_fit(fit_times, Ni_max=1, Nr_max=2, target_rsme=None)
print(fit_info['summary'])




HEOM_corr_fit = HEOMSolver(QobjEvo(H), (exp_bath, qt.sigmax()), max_depth=max_depth,
                           options={"nsteps": 15000, "rtol": 1e-12, "atol": 1e-12})
results_corr_fit = HEOM_corr_fit.run(psi0 * psi0.dag(), tlist, e_ops=e_ops)


# --- brmesolve ---

a_ops = [[qt.sigmax(), lambda w: exp_bath.power_spectrum(w)]]
brme_result = qt.brmesolve(H, psi0, tlist, a_ops=a_ops, e_ops=e_ops)

a_ops = [[qt.sigmax(), power_spectrum]]
brme_result2 = qt.brmesolve(H, psi0, tlist, a_ops=a_ops, e_ops=e_ops)


# --- Plots ---

plt.figure()

plt.plot(tlist, me_result.expect[0], '-', label=r'mesolve')
plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
plt.plot(
    tlist, brme_result.expect[0],
    ':', linewidth=6, label=r'brmesolve non-flat')
plt.plot(tlist, brme_result2.expect[0], ':', linewidth=6, label=r'brmesolve')

plt.xlabel(r'$t\, /\, \Delta^{-1}$', fontsize=18)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=18)
plt.legend()
plt.text(8, 0.7, "(b)", fontsize=18)

plt.savefig("mesolve_driven_2.pdf")

############ Alternative ways of specifying time dependence ############

f = f"sin({omega_d} * t)"

f = np.sin(omega_d * tlist)