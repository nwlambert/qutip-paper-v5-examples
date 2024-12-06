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

print(H)
#?\pythonoutput?Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper',
#?\pythonoutput?                dtype=CSR, isherm=True
#?\pythonoutput?Qobj data =
#?\pythonoutput?[[ 1.   0.   0.   0.1]
#?\pythonoutput? [ 0.   0.   0.1  0. ]
#?\pythonoutput? [ 0.   0.1  0.   0. ]
#?\pythonoutput? [ 0.1  0.   0.  -1. ]]

psi0 = qt.basis(2, 0) & qt.basis(2, 1)
tlist = np.linspace(0, 40, 100)
result = qt.sesolve(H, psi0, tlist)

solver = qt.SESolver(H)
result2 = solver.run(psi0, tlist)

t = 0.
dt = 40. / 100.
solver.start(psi0, t)
while t < 40:
    t = t + dt
    psi = solver.step(t)
    #dt = ... # process the result psi and calculate the next time step

options = {"store_states": True, "atol": 1e-12, "nsteps": 1000, "max_step": 0.1}
solver.options = options
result3 = solver.run(psi0, tlist)  # Or: qt.sesolve(H, psi0, tlist, options=options)
print(result3)
#?\pythonoutput?<Result
#?\pythonoutput?  Solver: sesolve
#?\pythonoutput?  Solver stats:
#?\pythonoutput?    method: 'scipy zvode adams'
#?\pythonoutput?    init time: 3.5762786865234375e-05
#?\pythonoutput?    preparation time: 0.0001609325408935547
#?\pythonoutput?    run time: 0.007315397262573242
#?\pythonoutput?    solver: 'Schrodinger Evolution'
#?\pythonoutput?  Time interval: [0.0, 40.0] (100 steps)
#?\pythonoutput?  Number of e_ops: 0
#?\pythonoutput?  States saved.
#?\pythonoutput?>