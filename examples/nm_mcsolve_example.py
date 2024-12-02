from mpl_setup import ORANGE, BLUE, GREEN, GRAY

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import qutip as qt
import matplotlib.pyplot as plt



# ----- SYSTEM PARAMETERS -----

# --- Numerics ---

tlist = np.linspace(0, 5, 500)

# --- Initial state ---

alphap = 1 / np.sqrt(2)
alpham = np.sqrt(1 - alphap**2)

initial_state = alphap * qt.basis(2, 0) + alpham * qt.basis(2, 1)

# --- Constants ---

gamma0 = 1
lamb = 0.3 * gamma0
Delta = 8 * lamb

# --- Derived quantities ---

delta = np.sqrt((lamb - 1j * Delta)**2 - 2 * gamma0 * lamb)
deltaR = np.real(delta)
deltaI = np.imag(delta)
deltaSq = deltaR**2 + deltaI**2

def TCL_params(t):
    common_factor = 2 * gamma0 * lamb / (
        (lamb**2 + Delta**2 - deltaSq) * np.cos(deltaI * t)
        - (lamb**2 + Delta**2 + deltaSq) * np.cosh(deltaR * t)
        - 2 * (Delta * deltaR + lamb * deltaI) * np.sin(deltaI * t)
        + 2 * (Delta * deltaI - lamb * deltaR) * np.sinh(deltaR * t)
    )
    gamma = common_factor * (
        lamb * np.cos(deltaI * t) - lamb * np.cosh(deltaR * t)
        - deltaI * np.sin(deltaI * t) - deltaR * np.sinh(deltaR * t)
    )
    A = common_factor * (
        Delta * np.cos(deltaI * t) - Delta * np.cosh(deltaR * t)
        - deltaR * np.sin(deltaI * t) + deltaI * np.sinh(deltaR * t)
    )
    return gamma, A

# --- Interpolate TCL params for better performance ---

_gamma = np.zeros_like(tlist)
_A = np.zeros_like(tlist)
for i in range(len(tlist)):
    _gamma[i], _A[i] = TCL_params(tlist[i])

gamma = CubicSpline(tlist, np.complex128(_gamma))
A = CubicSpline(tlist, np.complex128(_A))



# ----- MESOLVE -----

H = qt.sigmap() * qt.sigmam() / 2
unitary_gen = qt.liouvillian(H)
dissipator = qt.lindblad_dissipator(qt.sigmam())

me_solution = qt.mesolve([[unitary_gen, A], [dissipator, gamma]], initial_state, tlist)



# ----- NM_MCSOLVE -----

mc_solution = qt.nm_mcsolve([[H, A]], initial_state, tlist,
                            ops_and_rates=[(qt.sigmam(), gamma)],
                            ntraj=1_000, options={'map': 'parallel'}, seeds=0)



# ----- HEOM -----

# --- Schroedinger picture ---

# We are not in the interaction picture any more
# Choose, omega_c, omega_0 so that RWA is good
omega_c = 100
omega_0 = omega_c + Delta

H = omega_0 * qt.sigmap() * qt.sigmam()
Q = qt.sigmap() + qt.sigmam()

# --- Construct HEOM rhs ---

ck_real = [gamma0 * lamb / 4] * 2
vk_real = [lamb - 1j * omega_c, lamb + 1j * omega_c]
ck_imag = np.array([1j, -1j]) * gamma0 * lamb / 4
vk_imag = vk_real

heom_bath = qt.heom.BosonicBath(Q, ck_real, vk_real, ck_imag, vk_imag)

# --- Go ---

heom_solution = qt.heom.heomsolve(H, heom_bath, 10, qt.ket2dm(initial_state), tlist)

# --- Transform to interaction picture ---

Us = [(-1j * H * t).expm() for t in tlist]
heom_states = [U * state * U.dag() for (U, state) in zip(Us, heom_solution.states)]



# ----- BLOCH REDFIELD -----

def power_spectrum(w):
    return gamma0 * lamb**2 / ((omega_c - w)**2 + lamb**2)

br_solution = qt.brmesolve(H, initial_state, tlist, a_ops=[(qt.sigmax(), power_spectrum)])
br_states = [U * state * U.dag() for (U, state) in zip(Us, br_solution.states)]



# ----- PLOTS -----

# zeroes of gamma(t)
root1 = root_scalar(lambda t: TCL_params(t)[0], method='bisect', bracket=(1, 2)).root
root2 = root_scalar(lambda t: TCL_params(t)[0], method='bisect', bracket=(2, 3)).root
root3 = root_scalar(lambda t: TCL_params(t)[0], method='bisect', bracket=(3, 4)).root
root4 = root_scalar(lambda t: TCL_params(t)[0], method='bisect', bracket=(4, 5)).root

fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.54))

# --- rho_11 ---

projector = (qt.sigmaz() + qt.qeye(2)) / 2
ax[0].plot(tlist, qt.expect(projector, me_solution.states),
           '-', color=ORANGE, label=r'mesolve')
ax[0].plot(tlist[::10], qt.expect(projector, mc_solution.states[::10]),
           'x', color=BLUE, label=r'nm_mcsolve')
ax[0].plot(tlist, qt.expect(projector, heom_states),
           '--', color=GREEN, label=r'heomsolve')
ax[0].plot(tlist, qt.expect(projector, br_states),
           '-.', color=GRAY, label=r'brmesolve')

ax[0].set_xlabel(r'$t\, /\, \lambda^{-1}$', fontsize=18)
ax[0].set_xlim((-0.2, 5.2))
ax[0].set_xticks([0, 2.5, 5], labels=[r'$0$', r'$2.5$', r'$5$'])
ax[0].set_title(r'$\rho_{11}$', fontsize=18, pad=10)
ax[0].set_ylim((0.4376, 0.5024))
ax[0].set_yticks([0.44, 0.46, 0.48, 0.5], labels=[r'$0.44$', r'$0.46$', r'$0.48$', r'$0.50$'])

ax[0].axvspan(root1, root2, color=GRAY, alpha=0.08, zorder=0)
ax[0].axvspan(root3, root4, color=GRAY, alpha=0.08, zorder=0)

ax[0].legend(frameon=1, framealpha=1)

# --- | rho_01 |^2 ---

me_x = qt.expect(qt.sigmax(), me_solution.states)
mc_x = qt.expect(qt.sigmax(), mc_solution.states[::10])
heom_x = qt.expect(qt.sigmax(), heom_states)
br_x = qt.expect(qt.sigmax(), br_states)

me_y = qt.expect(qt.sigmay(), me_solution.states)
mc_y = qt.expect(qt.sigmay(), mc_solution.states[::10])
heom_y = qt.expect(qt.sigmay(), heom_states)
br_y = qt.expect(qt.sigmay(), br_states)

# The HEOM result oscillates with a very small amplitude that isn't
# visible in the plot but makes the dashing wonky. Let's smoothen.
heom_plot = heom_x * heom_x + heom_y * heom_y
heom_plot = np.convolve(heom_plot, np.array([1 / 11] * 11), mode='valid')
heom_tlist = tlist[5 : -5]

ax[1].plot(tlist, me_x * me_x + me_y * me_y,
           '-', color=ORANGE, label=r'mesolve')
ax[1].plot(tlist[::10], mc_x * mc_x + mc_y * mc_y,
           'x', color=BLUE, label=r'nm_mcsolve')
ax[1].plot(heom_tlist, heom_plot,
           '--', color=GREEN, label=r'heomsolve')
ax[1].plot(tlist, br_x * br_x + br_y * br_y,
           '-.', color=GRAY, label=r'brmesolve')

ax[1].set_xlabel(r'$t\, /\, \lambda^{-1}$', fontsize=18)
ax[1].set_xlim((-0.2, 5.2))
ax[1].set_xticks([0, 2.5, 5], labels=[r'$0$', r'$2.5$', r'$5$'])
ax[1].set_title(r'$| \rho_{01} |^2$', fontsize=18, pad=10)
ax[1].set_ylim((0.8752, 1.0048))
ax[1].set_yticks([0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1],
                 labels=[r'$0.88$', r'$0.90$', r'$0.92$', r'$0.94$', r'$0.96$', r'$0.98$', r'$1$'])

ax[1].axvspan(root1, root2, color=GRAY, alpha=0.08, zorder=0)
ax[1].axvspan(root3, root4, color=GRAY, alpha=0.08, zorder=0)

ax[1].legend(frameon=1, framealpha=1)

# --- Martingale ---

ax[2].plot(tlist, np.zeros_like(tlist), '-', color=ORANGE, label=r'Zero')
ax[2].plot(tlist[::10], 1000 * (mc_solution.trace[::10] - 1), 'x', color=BLUE, label=r'nm_mcsolve')

ax[2].set_xlabel(r'$t\, /\, \lambda^{-1}$', fontsize=18)
ax[2].set_xlim((-0.2, 5.2))
ax[2].set_xticks([0, 2.5, 5], labels=[r'$0$', r'$2.5$', r'$5$'])
ax[2].set_title(r'$(\mu - 1)\, /\, 10^{-3}$', fontsize=18, pad=10)
ax[2].set_ylim((-5.8, 15.8))
ax[2].set_yticks([-5, 0, 5, 10, 15])

ax[2].axvspan(root1, root2, color=GRAY, alpha=0.08, zorder=0)
ax[2].axvspan(root3, root4, color=GRAY, alpha=0.08, zorder=0)

ax[2].legend(frameon=1, framealpha=1)

plt.savefig('./nm_mcsolve_figure.pdf')



exit()
# ----- FORMATTING FOR PAPER -----

H = qt.sigmap() * qt.sigmam() / 2
initial_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit() 
tlist = np.linspace(0, 5, 500)



unitary_gen = qt.liouvillian(H)
dissipator = qt.lindblad_dissipator(qt.sigmam())
me_solution = qt.mesolve([[unitary_gen, A], [dissipator, gamma]], initial_state, tlist)



mc_solution = qt.nm_mcsolve([[H, A]], initial_state, tlist,
                            ops_and_rates=[(qt.sigmam(), gamma)],
                            ntraj=1_000, options={'map': 'parallel'})



H = omega_0 * qt.sigmap() * qt.sigmam()
Q = qt.sigmap() + qt.sigmam()



ck_real = [gamma0 * lamb / 4] * 2
vk_real = [lamb - 1j * omega_c, lamb + 1j * omega_c]
ck_imag = np.array([1j, -1j]) * gamma0 * lamb / 4
vk_imag = vk_real

heom_bath = qt.heom.BosonicBath(Q, ck_real, vk_real, ck_imag, vk_imag)
heom_solution = qt.heom.heomsolve(H, heom_bath, 10, qt.ket2dm(initial_state), tlist)



def power_spectrum(w):
    return gamma0 * lamb**2 / ((omega_c - w)**2 + lamb**2)

br_solution = qt.brmesolve(H, initial_state, tlist, a_ops=[(qt.sigmax(), power_spectrum)])



Us = [(-1j * H * t).expm() for t in tlist]
heom_states = [U * state * U.dag() for (U, state) in zip(Us, heom_solution.states)]
br_states = [U * state * U.dag() for (U, state) in zip(Us, br_solution.states)]