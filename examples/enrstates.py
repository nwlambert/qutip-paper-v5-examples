import qutip as qt
import numpy as np

eps = 2 * np.pi
omega_c = 2 * np.pi
g = 0.1 * omega_c
gam = 0.01 * omega_c 
tlist = np.linspace(0,20,100)


N_cut = 2

psi0 = qt.basis(2, 0) & qt.basis(N_cut, 0)
sz = qt.sigmaz() & qt.qeye(N_cut)
sm = qt.sigmam() & qt.qeye(N_cut)
a = qt.qeye(2) & qt.destroy(N_cut)

H_JC = (
    0.5 * eps * sz + omega_c * a.dag() * a +
    g * (a * sm.dag() + a.dag() * sm)
)

c_ops = [np.sqrt(gam) * a]

result_JC = qt.mesolve(H_JC, psi0, tlist, c_ops, e_ops=[sz])


N_exc = 1
dims = [2, N_cut]

psi0 = qt.enr_fock(dims, N_exc, [1, 0])
sm, a = qt.enr_destroy(dims, N_exc)
sz = 2 * sm.dag() * sm - 1

H_enr = (
    0.5 * eps * sz + omega_c * a.dag() * a +
    g * (a * sm.dag() + a.dag() * sm)
)

c_ops = [np.sqrt(gam) * a]

result_enr = qt.mesolve(H_enr, psi0, tlist, c_ops, e_ops=[sz]) 
    