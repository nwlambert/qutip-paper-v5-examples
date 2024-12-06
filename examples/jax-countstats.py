import qutip as qt
import jax
import jax.numpy as jnp
import qutip_jax  # noqa : F401

from diffrax import Tsit5, PIDController as PID

options = {"method": "diffrax",
           "normalize_output": False,
           "stepsize_controller": PID(rtol=1e-7, atol=1e-7),
           "solver": Tsit5(scan_kind="bounded"),
           "progress_bar": False}


ed = 1
GammaL = 1
GammaR = 1

with jax.default_device(jax.devices("cpu")[0]):
    with qt.CoreOptions(default_dtype="jaxdia"):       
        d = qt.destroy(2)
        H = ed * d.dag() * d
        c_op_L = jnp.sqrt(GammaL) * d.dag()
        c_op_R = jnp.sqrt(GammaR) * d

        L0 = (
            qt.liouvillian(H) + qt.lindblad_dissipator(c_op_L)
            - 0.5 * qt.spre(c_op_R.dag() * c_op_R)
            - 0.5 * qt.spost(c_op_R.dag() * c_op_R)
        )
        L1 = qt.sprepost(c_op_R, c_op_R.dag())

        
        rho0 = qt.steadystate(L0 + L1)
        

        def rhoz(t, z):
            L = L0 + jnp.exp(z) * L1  # jump term
            tlist = jnp.linspace(0, t, 50)     
            result = qt.mesolve(L, rho0, tlist, options=options)
            return result.final_state.tr()

        # first derivative
        drhozdz = jax.jacrev(rhoz, argnums=1) 
        # second derivative
        d2rhozdz = jax.jacfwd(drhozdz, argnums=1) 


tf = 100
Itest = GammaL * GammaR / (GammaL + GammaR)
print("Analytic current", Itest)
print("Numerical current", drhozdz(tf, 0.) / tf)
print("Analytical shot noise (2nd cumulant)",
      Itest * (1 - 2 * GammaL * GammaR / (GammaL + GammaR)**2))
print("Numerical shot noise (2nd cumulant)",
      (d2rhozdz(tf, 0.) - drhozdz(tf, 0.)**2) / tf)

#?\pythonoutput?Analytic current 0.5
#?\pythonoutput?Numerical current 0.4999
#?\pythonoutput?Analytical shot noise (2nd cumulant) 0.25
#?\pythonoutput?Numerical shot noise (2nd cumulant) 0.25125

