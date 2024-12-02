# ---IMPORT SECTION---
from mpl_setup import BLUE, ORANGE, GREEN, PURPLE, GRAY, SKY_BLUE, VERMILLION
from qutip import Qobj, sigmaz, sigmax, brmesolve, sigmay
from qutip import sigmam, sigmap, mesolve, expect, basis
import numpy as np
from qutip.solver.heom import HEOMSolver, UnderDampedBath

import matplotlib.pyplot as plt
import mpmath as mp
from environment import OhmicEnvironment, UnderDampedEnvironment, BosonicEnvironment
from environment import n_thermal
colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY, SKY_BLUE, VERMILLION]
P11p = basis(2, 0) * basis(2, 0).dag()
P12p = basis(2, 0) * basis(2, 1).dag()

# -- FIRST EXAMPLE PARAMETERS ---

lam = 0.5
w0 = 1
gamma = 0.1
T = 0.5
t = np.linspace(0, 40, 1000)
rho0 = Qobj([[0, 0], [0, 1]])
Hsys = sigmax()/2 + 3*sigmaz()/4
Q = sigmaz()
env = UnderDampedEnvironment(lam=lam, gamma=gamma, T=T, w0=w0)
bath = env.approx_by_matsubara(Nk=5).to_bath(Q)

# -- SIMULATION --

# -- HEOM --
solver = HEOMSolver(Hsys, bath, max_depth=9)
result_h = solver.run(rho0, t)
# APPROPIATE JUMP OPS IN THIS CASE
sp=Qobj([[ 0.15384615 , 0.04658087],[-0.50811933 ,-0.15384615]])
sz= Qobj([[ 0.69230769 , 0.46153846],[ 0.46153846, -0.69230769]])
sm = sp.dag()
# -- LINDBLAD --
c_ops = [np.sqrt(env.power_spectrum(1.803))*sp,
         np.sqrt(env.power_spectrum(-1.803))*sm,
         np.sqrt(env.power_spectrum(0))*sz]
result_lindblad = mesolve(Hsys, rho0, t, c_ops)


def nth(w):
    if T > 0:
        return 1 / (np.exp(w / T) - 1)
    else:
        return 0


def power_spectrum(w):
    if w > 0:
        return env.power_spectrum(w)
    elif w == 0:
        return 0
    else:
        return env.power_spectrum(-w)


# -- BLOCH-REDFIELD --
a_ops = [[Q, lambda w: env.power_spectrum(w).item()]]
resultBR = brmesolve(Hsys, rho0, t, a_ops=a_ops, sec_cutoff=-1)


# -- DYNAMICS SIMULATION --
fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.54))
ax[0].plot(lam*t, expect(P11p, result_h.states),
           color=colors[0], label='HEOMSolver',zorder=3)
ax[0].plot(lam*t, expect(P11p, result_lindblad.states),
            color=colors[1], label='mesolve')
ax[0].plot(lam*t, expect(P11p, resultBR.states),
           color=colors[2],linestyle=(0, (1, 1)), label='brmesolve')
ax[0].set_ylabel(r"$\rho_{11}$")
ax[0].set_xlabel(r"$\lambda t$")
ax[0].legend()
ax[1].plot(lam*t, expect(sigmap(), result_h.states),
           color=colors[0], label='HEOMSolver',zorder=3)
ax[1].plot(lam*t, expect(sigmap(), result_lindblad.states),
        color=colors[1], label='mesolve')
ax[1].plot(lam*t, expect(sigmap(), resultBR.states),
           color=colors[2],linestyle=(0, (1, 1)), label='brmesolve', markevery=10)

ax[1].set_ylabel(r"$Re(\rho_{12})$")
ax[1].set_xlabel(r"$\lambda t$")
ax[1].legend()
plt.savefig('./heom_qubit_underdamped.pdf')

# -- SECOND EXAMPLE PARAMETERS --
lam = 0.1
gamma = 5
T = 1

oh = OhmicEnvironment(T=T, alpha=lam, wc=gamma, s=1)

# --FITTING USING OHMIC CLASS--
w = np.linspace(0, 100, 2000)
env_fs, _ = oh.approx_by_sd_fit(wlist=w, Nk=3, Nmax=8)
bath_fs = env_fs.to_bath(Q)

t = np.linspace(0, 10, 1000)
env_fc, _ = oh.approx_by_cf_fit(tlist=t, Ni_max=5, Nr_max=4, target_rsme=None)
bath_fc = env_fc.to_bath(Q)

# -- USING A USER DEFINED ENVIRONMENT --


def J(w, lam=lam, gamma=gamma):
    """ 
    Ohmic spectral density
    """
    return lam*w*np.exp(-abs(w)/gamma)


user_env = BosonicEnvironment.from_spectral_density(J, T=T, wMax=60)

user_env_sd, _ = user_env.approx_by_sd_fit(wlist=w, Nk=3, Nmax=8)
bath_env_sd = user_env_sd.to_bath(Q)
user_env_cf, _ = user_env.approx_by_cf_fit(tlist=t, Ni_max=5, Nr_max=4)
bath_env_cf = user_env_cf.to_bath(Q)


# -- SOLVING DYNAMICS --
tlist = np.linspace(0, 10, 1000)
HEOM_corr_fit = HEOMSolver(Hsys, bath_fc, max_depth=5)
result_corr = HEOM_corr_fit.run(rho0, tlist)

HEOM_spec_fit = HEOMSolver(Hsys, bath_fs, max_depth=5)
result_spec = HEOM_spec_fit.run(rho0, tlist)


HEOM_fos = HEOMSolver(Hsys, bath_env_sd,
                      max_depth=5)
result_fos = HEOM_fos.run(rho0, tlist)

HEOM_foc = HEOMSolver(Hsys, bath_env_cf,
                      max_depth=5)
result_foc = HEOM_foc.run(rho0, tlist)

# -- OHMIC BATH DYNAMICS --
figfit, axfit = plt.subplots(1, 2, figsize=(13.6, 4.54))
full = env_fs.spectral_density(w)
axfit[0].plot(w, full, color=colors[0], label="Original")
j = 1
markers = ["-.", "--", "-."]
for i in [1, 10, 15]:
    bath_fs, fitinfo = env_fs.approx_by_sd_fit(wlist=w,Nmax=8, Nk=i,target_rsme=None)
    print(fitinfo["summary"])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    axfit[0].plot(w, bath_fs.spectral_density(
        w), linestyle=markers[j-1], color=colors[j], label=f"k={i}")
    axfit[1].plot(w, np.abs(full-bath_fs.spectral_density(w)),
                  markers[j-1], color=colors[j], label=f"k={i}")
    j += 1

axfit[0].legend()
axfit[1].legend()
axfit[0].set_ylabel(r"$J(\omega)$")
axfit[0].set_xlabel(r"$\omega$")
axfit[1].set_ylabel(r"$|J(\omega)-J_{approx}(\omega)|$")
axfit[1].set_xlabel(r"$\omega$")
plt.savefig('./heom_spec_k.pdf')

figfit, axfit = plt.subplots(1, 2, figsize=(13.6, 4.54))
full = env_fs.correlation_function(t)
full_sd = env_fs.spectral_density(w)

# EXAMPLE OF COMPUTING APPROX CORRELATION FUNCTION
env = UnderDampedEnvironment(lam=lam, gamma=gamma, T=T, w0=w0)
bath = env.approx_by_matsubara(Nk=5)
C = env.correlation_function(t)
C2 = bath.correlation_function(t)
