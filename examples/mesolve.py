import qutip as qt
import numpy as np

epsilon1 = 1.0
epsilon2 = 1.0
g = 0.1

sx1 = qt.sigmax() & qt.qeye(2)
sx2 = qt.qeye(2) & qt.sigmax()
sz1 = qt.sigmaz() & qt.qeye(2)
sz2 = qt.qeye(2) & qt.sigmaz()

H = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2

sm1 = qt.sigmam() & qt.qeye(2)
sm2 = qt.qeye(2) & qt.sigmam()
gam = 0.1  # dissipation rate
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]
psi0 = qt.basis(2, 0) & qt.basis(2, 1)
tlist = np.linspace(0, 40, 100)

result_me = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[sz1, sz2])
print(result_me)
#?\pythonoutput?<Result
#?\pythonoutput?  Solver: mesolve
#?\pythonoutput?  Solver stats:
#?\pythonoutput?    method: 'scipy zvode adams'
#?\pythonoutput?    init time: 8.153915405273438e-05
#?\pythonoutput?    preparation time: 0.00011157989501953125
#?\pythonoutput?    run time: 0.0024967193603515625
#?\pythonoutput?    solver: 'Master Equation Evolution'
#?\pythonoutput?    num_collapse: 2
#?\pythonoutput?  Time interval: [0.0, 40.0] (100 steps)
#?\pythonoutput?  Number of e_ops: 2
#?\pythonoutput?  State not saved.
#?\pythonoutput?>

lindbladian = -1.0j * (qt.spre(H) - qt.spost(H)) 
for c in c_ops:
    lindbladian += (
        qt.sprepost(c, c.dag()) - 0.5 * (qt.spre(c.dag() * c) + qt.spost(c.dag() * c))
    )

result_me2 = qt.mesolve(lindbladian, psi0, tlist,  [], e_ops = [sz1, sz2])


def power_spectrum(w):
    if w >= 0:
        return gam
    else:
        return 0

all_energy, all_state = H.eigenstates()
Nmax = len(all_state)
collapse_list = []
for i in range(Nmax):
    for j in range(Nmax):
        delE = (all_energy[j] - all_energy[i])
        rate = power_spectrum(delE) * (
            np.absolute(sx1.matrix_element(all_state[i].dag(), all_state[j])) ** 2
            + np.absolute(sx2.matrix_element(all_state[i].dag(), all_state[j])) ** 2
        )
        if rate > 0:
            collapse_list.append(np.sqrt(rate) * all_state[i] * all_state[j].dag())

tlist_long = np.linspace(0, 1000, 100)
result_me_global = qt.mesolve(H, psi0, tlist_long, collapse_list)
fidelity = qt.fidelity(result_me_global.states[-1], all_state[0] @ all_state[0].dag())
print(f"Fidelity with ground-state: {fidelity:.6f}")
#?\pythonoutput?Fidelity with ground-state: 1.000000




def power_spectrum(w):
    if w >= 0:
        return gam
    else:
        return 0

result_BR = qt.brmesolve(H, psi0, tlist, e_ops=[sz1, sz2],
                         a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]])

# repeat mesolve simulation with different tlist
result_me_global = qt.mesolve(H, psi0, tlist, collapse_list, e_ops = [sz1, sz2])

import matplotlib.pyplot as plt
from mpl_setup import * 
plt.figure()
plt.plot(tlist, result_me.expect[0], label=r'Local Lindblad')
plt.plot(tlist, result_me_global.expect[0],'--',label=r'Dressed Lindblad')
plt.plot(tlist, result_BR.expect[0],':',label=r'Bloch-Redfield')
plt.xlabel('Time', fontsize=18)
plt.ylabel(r'$\langle \sigma_z^{(1)} \rangle$', fontsize=18)
plt.legend()
plt.text(10,0.7,"(a)", fontsize=18)
#plt.savefig("mesolve-fig1.pdf")
plt.show()
