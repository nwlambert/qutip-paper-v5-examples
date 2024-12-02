# QuTiP 5 paper examples
A selection of examples released with the QuTiP 5 paper,  available on the ArXiV at...

These examples are provided in a unmaintained static format, alongside a environment requirements file for future reproducability. 
Continously maintained and documented versions of the main examples are available in the qutip tutorials repository.


If you use Anaconda, you can install the dependencies for this repository with

```shell
cd qutip-paper-v5-examples
conda env create --file environment.yml
conda activate qutip-paper
```

Alternatively, with `pip`

```shell
pip install -r requirements.txt
```

List of files and their purpose:

enr_waveguide.py:  An example of how to use ENR states to simulate a qubit emitting into a waveguide.
enrstates.py:  Basic example of ENR states
jax-countstats.py: An example of how to use qutip-jax + jax Auto-diff to calculate counting statistics from a master equation.
jax-mesolve.py: Basic setup of an Ising spin chain model, used as a foundation for the GPU benchmark results in the QuTiP 5 paper.
mcsolve.py: An example of how to QuTiP's Monte-Carlo quantum jump solver.
mesolve.py: An example of how to use QuTiP's master equation, both directly and with globally-defined jump rates, and  a comparison to the Bloch-Redfield solver.
mpl_setup.py: Utility file for consisten matlpotlib settings.
nm_mcsolve_example.py: An example of how to use the non-Markovia Monte-Carlo solver (for a qubit coupled to a waveguide)
p_Floquet_speed_text.py: Benchmarking of Floquet states versus direct sesolve().
p_Ising_with_drive_Floquet.py: Benchmarking of Floquet states versus system size with the Ising model.
qip_renderer.py: Basic example of new QuTiP-QIP circuit plotting functionality.
qoc.py: Example of new optimal control methods from qutip-qoc.
qutipqip.py: A more involved example of how to use qutip-qip for quantum simulation.
smesolve.py: A basic example of how to use the stochastic master equation solver.
solver.py: A basic example of how to use the new solver interface.
/environment_examples/environment.py: A utility file for the new environment class interface. Replaced by latest official qutip release.
/environment_examples/heom_example.py: A basic example of how to use the new envirionment interface with the HEOM solver.
/environment_examples/heom_transition.py:  A more involved example of how to use the fitting features to capture the spin-boson phase transition with the HEOM solver.
/environment_examples/mesolvedriven.py: An example of how to model driven systems with mesolve and the HEOM solver.