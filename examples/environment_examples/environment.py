from __future__ import annotations
import abc
import enum
from time import time
from typing import Any, Callable, Sequence
import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvalsh
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
try:
    from mpmath import mp
    _mpmath_available = True
except ModuleNotFoundError:
    _mpmath_available = False

from qutip import spre, spost, Qobj
from qutip.solver.heom import BathExponent, BosonicBath


class BosonicEnvironment(abc.ABC):
    """
    The bosonic environment of an open quantum system. It is characterized by
    its spectral density and temperature or, equivalently, its power spectrum
    or its two-time auto-correlation function.

    Use one of the classmethods :meth:`from_spectral_density`,
    :meth:`from_power_spectrum` or :meth:`from_correlation_function` to
    construct an environment manually from one of these characteristic
    functions, or use a predefined sub-class such as the
    :class:`DrudeLorentzEnvironment`, the :class:`UnderDampedEnvironment` or
    the :class:`OhmicEnvironment`.

    Bosonic environments offer various ways to approximate the environment with
    a multi-exponential correlation function, which can be used for example in
    the HEOM solver. The approximated environment is represented as a
    :class:`ExponentialBosonicEnvironment`.

    All bosonic environments can be approximated by directly fitting their
    correlation function with a multi-exponential ansatz
    (:meth:`approx_by_cf_fit`) or by fitting their spectral density with a sum
    of Lorentzians (:meth:`approx_by_sd_fit`), which correspond to
    underdamped environments with known multi-exponential decompositions.
    Subclasses may offer additional approximation methods such as
    :meth:`DrudeLorentzEnvironment.approx_by_matsubara` or
    :meth:`DrudeLorentzEnvironment.approx_by_pade` in the case of a
    Drude-Lorentz environment.

    Parameters
    ----------
    T : optional, float
        The temperature of this environment.
    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(self, T: float = None, tag: Any = None):
        self.T = T
        self.tag = tag

    @abc.abstractmethod
    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        The spectral density of this environment. For negative frequencies,
        a value of zero will be returned. See the Users Guide on
        :ref:`bosonic environments <bosonic environments guide>` for specifics
        on the definitions used by QuTiP.

        If no analytical expression for the spectral density is known, it will
        be derived from the power spectrum. In this case, the temperature of
        this environment must be specified.

        If no analytical expression for the power spectrum is known either, it
        will be derived from the correlation function via a fast fourier
        transform.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the spectral density.
        """

        ...

    @abc.abstractmethod
    def correlation_function(
        self, t: float | ArrayLike, *, eps: float = 1e-10
    ) -> (float | ArrayLike):
        """
        The two-time auto-correlation function of this environment. See the
        Users Guide on :ref:`bosonic environments <bosonic environments guide>`
        for specifics on the definitions used by QuTiP.

        If no analytical expression for the correlation function is known, it
        will be derived from the power spectrum via a fast fourier transform.

        If no analytical expression for the power spectrum is known either, it
        will be derived from the spectral density. In this case, the
        temperature of this environment must be specified.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.

        eps : optional, float
            Used in case the power spectrum is derived from the spectral
            density; see the documentation of
            :meth:`BosonicEnvironment.power_spectrum`.
        """

        ...

    @abc.abstractmethod
    def power_spectrum(
        self, w: float | ArrayLike, *, eps: float = 1e-10
    ) -> (float | ArrayLike):
        """
        The power spectrum of this environment. See the Users Guide on
        :ref:`bosonic environments <bosonic environments guide>` for specifics
        on the definitions used by QuTiP.

        If no analytical expression for the power spectrum is known, it will
        be derived from the spectral density. In this case, the temperature of
        this environment must be specified.

        If no analytical expression for the spectral density is known either,
        the power spectrum will instead be derived from the correlation
        function via a fast fourier transform.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the power spectrum.

        eps : optional, float
            To derive the zero-frequency power spectrum from the spectral
            density, the spectral density must be differentiated numerically.
            In that case, this parameter is used as the finite difference in
            the numerical differentiation.
        """

        ...

    # --- user-defined environment creation

    @classmethod
    def from_correlation_function(
        cls,
        C: Callable[[float], complex] | ArrayLike,
        tlist: ArrayLike = None,
        tMax: float = None,
        *,
        T: float = None,
        tag: Any = None
    ) -> BosonicEnvironment:
        r"""
        Constructs a bosonic environment from the provided correlation
        function. The provided function will only be used for times
        :math:`t \geq 0`. At times :math:`t < 0`, the symmetry relation
        :math:`C(-t) = C(t)^\ast` is enforced.

        Parameters
        ----------
        C : callable or array_like
            The correlation function.

        tlist : optional, array_like
            The times where the correlation function is sampled (if it is
            provided as an array).

        tMax : optional, float
            Specifies that the correlation function is essentially zero outside
            the interval [-tMax, tMax]. Used for numerical integration
            purposes.

        T : optional, float
            Bath temperature. (The spectral density of this environment can
            only be calculated from the correlation function if a temperature
            is provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.
        """
        return _BosonicEnvironment_fromCF(C, tlist, tMax, T, tag)

    @classmethod
    def from_power_spectrum(
        cls,
        S: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        wMax: float = None,
        *,
        T: float = None,
        tag: Any = None
    ) -> BosonicEnvironment:
        """
        Constructs a bosonic environment with the provided power spectrum.

        Parameters
        ----------
        S : callable or array_like
            The power spectrum.

        wlist : optional, array_like
            The frequencies where the power spectrum is sampled (if it is
            provided as an array).

        wMax : optional, float
            Specifies that the power spectrum is essentially zero outside the
            interval [-wMax, wMax]. Used for numerical integration purposes.

        T : optional, float
            Bath temperature. (The spectral density of this environment can
            only be calculated from the power spectrum if a temperature
            is provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.
        """
        return _BosonicEnvironment_fromPS(S, wlist, wMax, T, tag)

    @classmethod
    def from_spectral_density(
        cls,
        J: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        wMax: float = None,
        *,
        T: float = None,
        tag: Any = None
    ) -> BosonicEnvironment:
        r"""
        Constructs a bosonic environment with the provided spectral density.
        The provided function will only be used for frequencies
        :math:`\omega > 0`. At frequencies :math:`\omega \leq 0`, the spectral
        density is zero according to the definition used by QuTiP. See the
        Users Guide on :ref:`bosonic environments <bosonic environments guide>`
        for a note on spectral densities with support at negative frequencies.

        Parameters
        ----------
        J : callable or array_like
            The spectral density.

        wlist : optional, array_like
            The frequencies where the spectral density is sampled (if it is
            provided as an array).

        wMax : optional, float
            Specifies that the spectral density is essentially zero outside the
            interval [-wMax, wMax]. Used for numerical integration purposes.

        T : optional, float
            Bath temperature. (The correlation function and the power spectrum
            of this environment can only be calculated from the spectral
            density if a temperature is provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.
        """
        return _BosonicEnvironment_fromSD(J, wlist, wMax, T, tag)

    # --- spectral density, power spectrum, correlation function conversions

    def _ps_from_sd(self, w, eps, derivative=None):
        # derivative: value of J'(0)
        if self.T is None:
            raise ValueError(
                "Bath temperature must be specified for this operation")

        w = np.array(w, dtype=float)
        if self.T == 0:
            return 2 * np.heaviside(w, 0) * self.spectral_density(w)

        # at zero frequency, we do numerical differentiation
        # S(0) = 2 J'(0) / beta
        zero_mask = (w == 0)
        nonzero_mask = np.invert(zero_mask)

        S = np.zeros_like(w)
        if derivative is None:
            S[zero_mask] = 2 * self.T * self.spectral_density(eps) / eps
        else:
            S[zero_mask] = 2 * self.T * derivative
        S[nonzero_mask] = (
            2 * np.sign(w[nonzero_mask])
            * self.spectral_density(np.abs(w[nonzero_mask]))
            * (n_thermal(w[nonzero_mask], self.T) + 1)
        )
        return S

    def _sd_from_ps(self, w):
        if self.T is None:
            raise ValueError(
                "Bath temperature must be specified for this operation")

        w = np.array(w, dtype=float)
        J = np.zeros_like(w)
        positive_mask = (w > 0)

        J[positive_mask] = (
            self.power_spectrum(w[positive_mask]) / 2
            / (n_thermal(w[positive_mask], self.T) + 1)
        )
        return J

    def _ps_from_cf(self, w, tMax):
        w = np.array(w, dtype=float)
        if w.ndim == 0:
            wMax = np.abs(w)
        else:
            wMax = max(np.abs(w[0]), np.abs(w[-1]))

        mirrored_result = _fft(self.correlation_function, wMax, tMax=tMax)
        return np.real(mirrored_result(-w))

    def _cf_from_ps(self, t, wMax, **ps_kwargs):
        t = np.array(t, dtype=float)
        if t.ndim == 0:
            tMax = np.abs(t)
        else:
            tMax = max(np.abs(t[0]), np.abs(t[-1]))

        result = _fft(lambda w: self.power_spectrum(w, **ps_kwargs),
                      tMax, tMax=wMax)
        return result(t) / (2 * np.pi)

    # --- fitting

    def approx_by_cf_fit(
        self, tlist: ArrayLike, target_rsme: float = 2e-5, Nr_max: int = 10,
        Ni_max: int = 10, guess: list[float] = None, lower: list[float] = None,
        upper: list[float] = None, full_ansatz: bool = False, tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        r"""
        Generates an approximation to this environment by fitting its
        correlation function with a multi-exponential ansatz. The number of
        exponents is determined iteratively based on reducing the normalized
        root mean squared error below a given threshold.

        Specifically, the real and imaginary parts are fit by the following
        model functions:

        .. math::
            \operatorname{Re}[C(t)] = \sum_{k=1}^{N_r} \operatorname{Re}\Bigl[
                (a_k + \mathrm i d_k) \mathrm e^{(b_k + \mathrm i c_k) t}\Bigl]
                ,
            \\
            \operatorname{Im}[C(t)] = \sum_{k=1}^{N_i} \operatorname{Im}\Bigl[
                (a'_k + \mathrm i d'_k) \mathrm e^{(b'_k + \mathrm i c'_k) t}
                \Bigr].

        If the parameter `full_ansatz` is `False`, :math:`d_k` and :math:`d'_k`
        are set to zero and the model functions simplify to

        .. math::
            \operatorname{Re}[C(t)] = \sum_{k=1}^{N_r}
                a_k  e^{b_k  t} \cos(c_{k} t)
                ,
            \\
            \operatorname{Im}[C(t)] = \sum_{k=1}^{N_i}
                a'_k  e^{b'_k  t} \sin(c'_{k} t) .

        The simplified version offers faster fits, however it fails for
        anomalous spectral densities with
        :math:`\operatorname{Im}[C(0)] \neq 0` as :math:`\sin(0) = 0`.



        Parameters
        ----------
        tlist : array_like
            The time range on which to perform the fit.
        target_rmse : optional, float
            Desired normalized root mean squared error (default `2e-5`). Can be
            set to `None` to perform only one fit using the maximum number of
            modes (`Nr_max`, `Ni_max`).
        Nr_max : optional, int
            The maximum number of modes to use for the fit of the real part
            (default 10).
        Ni_max : optional, int
            The maximum number of modes to use for the fit of the imaginary
            part (default 10).
        guess : optional, list of float
            Initial guesses for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same initial guesses are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `guess` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        lower : optional, list of float
            Lower bounds for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same lower bounds are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `lower` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        upper : optional, list of float
            Upper bounds for the parameters :math:`a_k`, :math:`b_k`, etc.
            The same upper bounds are used for all values of k, and for
            the real and imaginary parts. If `full_ansatz` is True, `upper` is
            a list of size 4, otherwise, it is a list of size 3.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        full_ansatz : optional, bool (default False)
            If this is set to False, the parameters :math:`d_k` are all set to
            zero. The full ansatz, including :math:`d_k`, usually leads to
            significantly slower fits, and some manual tuning of the `guesses`,
            `lower` and `upper` is usually needed. On the other hand, the full
            ansatz can lead to better fits with fewer exponents, especially
            for anomalous spectral densities with
            :math:`\operatorname{Im}[C(0)] \neq 0` for which the simplified
            ansatz will always give :math:`\operatorname{Im}[C(0)] = 0`.
            When using the full ansatz with default values for the guesses and
            bounds, if the fit takes too long, we recommend choosing guesses
            and bounds manually.
        tag : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

        Returns
        -------
        approx_env : :class:`ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.
        fit_info : dictionary
            A dictionary containing the following information about the fit.

            "Nr"
                The number of terms used to fit the real part of the
                correlation function.
            "Ni"
                The number of terms used to fit the imaginary part of the
                correlation function.
            "fit_time_real"
                The time the fit of the real part of the correlation function
                took in seconds.
            "fit_time_imag"
                The time the fit of the imaginary part of the correlation
                function took in seconds.
            "rsme_real"
                Normalized mean squared error obtained in the fit of the real
                part of the correlation function.
            "rsme_imag"
                Normalized mean squared error obtained in the fit of the
                imaginary part of the correlation function.
            "params_real"
                The fitted parameters (array of shape Nx3 or Nx4) for the real
                part of the correlation function.
            "params_imag"
                The fitted parameters (array of shape Nx3 or Nx4) for the
                imaginary part of the correlation function.
            "summary"
                A string that summarizes the information about the fit.
        """

        # Process arguments
        if tag is None and self.tag is not None:
            tag = (self.tag, "CF Fit")

        if full_ansatz:
            num_params = 4
        else:
            num_params = 3

        if target_rsme is None:
            target_rsme = 0
            Nr_min, Ni_min = Nr_max, Ni_max
        else:
            Nr_min, Ni_min = 1, 1

        clist = self.correlation_function(tlist)
        if guess is None and lower is None and upper is None:
            guess_re, lower_re, upper_re = _default_guess_cfreal(
                tlist, np.real(clist), full_ansatz)
            guess_im, lower_im, upper_im = _default_guess_cfimag(
                np.imag(clist), full_ansatz)
        else:
            guess_re, lower_re, upper_re = guess, lower, upper
            guess_im, lower_im, upper_im = guess, lower, upper

        # Fit real part
        start_real = time()
        rmse_real, params_real = iterated_fit(
            _cf_real_fit_model, num_params, tlist, np.real(clist), target_rsme,
            guess_re, Nr_min, Nr_max, lower_re, upper_re
        )
        end_real = time()
        fit_time_real = end_real - start_real

        # Fit imaginary part
        start_imag = time()
        rmse_imag, params_imag = iterated_fit(
            _cf_imag_fit_model, num_params, tlist, np.imag(clist), target_rsme,
            guess_im, Ni_min, Ni_max, lower_im, upper_im
        )
        end_imag = time()
        fit_time_imag = end_imag - start_imag

        # Generate summary
        Nr = len(params_real)
        Ni = len(params_imag)
        full_summary = _cf_fit_summary(
            params_real, params_imag, fit_time_real, fit_time_imag,
            Nr, Ni, rmse_real, rmse_imag, n=num_params
        )

        fit_info = {"Nr": Nr, "Ni": Ni, "fit_time_real": fit_time_real,
                    "fit_time_imag": fit_time_imag, "rmse_real": rmse_real,
                    "rmse_imag": rmse_imag, "params_real": params_real,
                    "params_imag": params_imag, "summary": full_summary}

        # Finally, generate environment and return
        ckAR = []
        vkAR = []
        for term in params_real:
            if full_ansatz:
                a, b, c, d = term
            else:
                a, b, c = term
                d = 0
            ckAR.extend([(a + 1j * d) / 2, (a - 1j * d) / 2])
            vkAR.extend([-b - 1j * c, -b + 1j * c])

        ckAI = []
        vkAI = []
        for term in params_imag:
            if full_ansatz:
                a, b, c, d = term
            else:
                a, b, c = term
                d = 0
            ckAI.extend([-1j * (a + 1j * d) / 2, 1j * (a - 1j * d) / 2])
            vkAI.extend([-b - 1j * c, -b + 1j * c])

        approx_env = ExponentialBosonicEnvironment(
            ckAR, vkAR, ckAI, vkAI, T=self.T, tag=tag)
        return approx_env, fit_info

    def approx_by_sd_fit(
        self, wlist: ArrayLike, Nk: int = 1, target_rsme: float = 5e-6,
        Nmax: int = 10, guess: list[float] = None, lower: list[float] = None,
        upper: list[float] = None, tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        r"""
        Generates an approximation to this environment by fitting its spectral
        density with a sum of underdamped terms. Each underdamped term
        effectively acts like an underdamped environment. We use the known
        exponential decomposition of the underdamped environment, keeping `Nk`
        Matsubara terms for each. The number of underdamped terms is determined
        iteratively based on reducing the normalized root mean squared error
        below a given threshold.

        Specifically, the spectral density is fit by the following model
        function:

        .. math::
            J(\omega) = \sum_{k=1}^{N} \frac{2 a_k b_k \omega}{\left(\left(
                \omega + c_k \right)^2 + b_k^2 \right) \left(\left(
                \omega - c_k \right)^2 + b_k^2 \right)}

        Parameters
        ----------
        wlist : array_like
            The frequency range on which to perform the fit.
        Nk : optional, int
            The number of Matsubara terms to keep in each mode (default 1).
        target_rmse : optional, float
            Desired normalized root mean squared error (default `5e-6`). Can be
            set to `None` to perform only one fit using the maximum number of
            modes (`Nmax`).
        Nmax : optional, int
            The maximum number of modes to use for the fit (default 10).
        guess : optional, list of float
            Initial guesses for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same initial guesses are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        lower : optional, list of float
            Lower bounds for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same lower bounds are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        upper : optional, list of float
            Upper bounds for the parameters :math:`a_k`, :math:`b_k` and
            :math:`c_k`. The same upper bounds are used for all values of
            k.
            If none of `guess`, `lower` and `upper` are provided, these
            parameters will be chosen automatically.
        tag : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

        Returns
        -------
        approx_env : :class:`ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.
        fit_info : dictionary
            A dictionary containing the following information about the fit.

            "N"
                The number of underdamped terms used in the fit.
            "Nk"
                The number of Matsubara modes included per underdamped term.
            "fit_time"
                The time the fit took in seconds.
            "rsme"
                Normalized mean squared error obtained in the fit.
            "params"
                The fitted parameters (array of shape Nx3).
            "summary"
                A string that summarizes the information about the fit.
        """

        # Process arguments
        if tag is None and self.tag is not None:
            tag = (self.tag, "SD Fit")

        if target_rsme is None:
            target_rsme = 0
            Nmin = Nmax
        else:
            Nmin = 1

        jlist = self.spectral_density(wlist)
        if guess is None and lower is None and upper is None:
            guess, lower, upper = _default_guess_sd(wlist, jlist)

        # Fit
        start = time()
        rmse, params = iterated_fit(
            _sd_fit_model, 3, wlist, jlist, target_rsme, guess,
            Nmin, Nmax, lower, upper
        )
        end = time()
        fit_time = end - start

        # Generate summary
        N = len(params)
        summary = _fit_summary(
            fit_time, rmse, N, "the spectral density", params
        )
        fit_info = {
            "N": N, "Nk": Nk, "fit_time": fit_time, "rmse": rmse,
            "params": params, "summary": summary}

        ckAR, vkAR, ckAI, vkAI = [], [], [], []
        # Finally, generate environment and return
        for a, b, c in params:
            lam = np.sqrt(a + 0j)
            gamma = 2 * b
            w0 = np.sqrt(c**2 + b**2)

            env = UnderDampedEnvironment(self.T, lam, gamma, w0)
            coeffs = env._matsubara_params(Nk)
            ckAR.extend(coeffs[0])
            vkAR.extend(coeffs[1])
            ckAI.extend(coeffs[2])
            vkAI.extend(coeffs[3])

        approx_env = ExponentialBosonicEnvironment(
            ckAR, vkAR, ckAI, vkAI, T=self.T, tag=tag)
        return approx_env, fit_info


class _BosonicEnvironment_fromCF(BosonicEnvironment):
    def __init__(self, C, tlist, tMax, T, tag):
        super().__init__(T, tag)
        self._cf = _complex_interpolation(C, tlist, 'correlation function')
        if tlist is not None:
            self.tMax = max(np.abs(tlist[0]), np.abs(tlist[-1]))
        else:
            self.tMax = tMax

    def correlation_function(self, t, **kwargs):
        t = np.array(t, dtype=float)
        result = np.zeros_like(t, dtype=complex)
        positive_mask = (t >= 0)
        non_positive_mask = np.invert(positive_mask)

        result[positive_mask] = self._cf(t[positive_mask])
        result[non_positive_mask] = np.conj(
            self._cf(-t[non_positive_mask])
        )
        return result

    def spectral_density(self, w):
        return self._sd_from_ps(w)

    def power_spectrum(self, w, **kwargs):
        if self.tMax is None:
            raise ValueError('The support of the correlation function (tMax) '
                             'must be specified in order to compute the power '
                             'spectrum.')
        return self._ps_from_cf(w, self.tMax)


class _BosonicEnvironment_fromPS(BosonicEnvironment):
    def __init__(self, S, wlist, wMax, T, tag):
        super().__init__(T, tag)
        self._ps = _real_interpolation(S, wlist, 'power spectrum')
        if wlist is not None:
            self.wMax = max(np.abs(wlist[0]), np.abs(wlist[-1]))
        else:
            self.wMax = wMax

    def correlation_function(self, t, **kwargs):
        if self.wMax is None:
            raise ValueError('The support of the power spectrum (wMax) '
                             'must be specified in order to compute the '
                             'correlation function.')
        return self._cf_from_ps(t, self.wMax)

    def spectral_density(self, w):
        return self._sd_from_ps(w)

    def power_spectrum(self, w, **kwargs):
        w = np.array(w, dtype=float)
        return self._ps(w)


class _BosonicEnvironment_fromSD(BosonicEnvironment):
    def __init__(self, J, wlist, wMax, T, tag):
        super().__init__(T, tag)
        self._sd = _real_interpolation(J, wlist, 'spectral density')
        if wlist is not None:
            self.wMax = max(np.abs(wlist[0]), np.abs(wlist[-1]))
        else:
            self.wMax = wMax

    def correlation_function(self, t, *, eps=1e-10):
        if self.wMax is None:
            raise ValueError('The support of the spectral density (wMax) '
                             'must be specified in order to compute the '
                             'correlation function.')
        return self._cf_from_ps(t, self.wMax, eps=eps)

    def spectral_density(self, w):
        w = np.array(w, dtype=float)

        result = np.zeros_like(w)
        positive_mask = (w > 0)
        result[positive_mask] = self._sd(w[positive_mask])

        return result

    def power_spectrum(self, w, *, eps=1e-10):
        return self._ps_from_sd(w, eps)


class UnderDampedEnvironment(BosonicEnvironment):
    r"""
    Describes an underdamped environment with the spectral density

    .. math::
        J(\omega) = \frac{\lambda^{2} \Gamma \omega}{(\omega_{c}^{2}-
        \omega^{2})^{2}+ \Gamma^{2} \omega^{2}}

    (see Eq. 16 in [BoFiN23]_).

    Parameters
    ----------
    T : float
        Bath temperature.

    lam : float
        Coupling strength.

    gamma : float
        Bath spectral density cutoff frequency.

    w0 : float
        Bath spectral density resonance frequency.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, lam: float, gamma: float, w0: float, *, tag: Any = None
    ):
        super().__init__(T, tag)

        self.lam = lam
        self.gamma = gamma
        self.w0 = w0

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the underdamped spectral density.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.array(w, dtype=float)
        result = np.zeros_like(w)

        positive_mask = (w > 0)
        w_mask = w[positive_mask]
        result[positive_mask] = (
            self.lam**2 * self.gamma * w_mask / (
                (w_mask**2 - self.w0**2)**2 + (self.gamma * w_mask)**2
            )
        )

        return result

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum of the underdamped environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        sd_derivative = self.lam**2 * self.gamma / self.w0**4
        return self._ps_from_sd(w, None, sd_derivative)

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the two-time auto-correlation function of the underdamped
        environment.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """
        # we need an wMax so that spectral density is zero for w>wMax, guess:
        wMax = self.w0 + 10 * self.gamma
        return self._cf_from_ps(t, wMax)

    def approx_by_matsubara(
        self, Nk: int, combine: bool = True, tag: Any = None
    ) -> ExponentialBosonicEnvironment:
        """
        Generates an approximation to this environment by truncating its
        Matsubara expansion.

        Parameters
        ----------
        Nk : int
            Number of Matsubara terms to include. In total, the real part of
            the correlation function will include `Nk+2` terms and the
            imaginary part `2` terms.

        combine : bool, default `True`
            Whether to combine exponents with the same frequency.

        tag : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

        Returns
        -------
        :class:`ExponentialBosonicEnvironment`
            The approximated environment with multi-exponential correlation
            function.
        """

        if tag is None and self.tag is not None:
            tag = (self.tag, "Matsubara Truncation")

        lists = self._matsubara_params(Nk)
        result = ExponentialBosonicEnvironment(
            *lists, T=self.T, combine=combine, tag=tag)
        return result

    def _matsubara_params(self, Nk):
        """ Calculate the Matsubara coefficients and frequencies. """
        beta = 1 / self.T
        Om = np.sqrt(self.w0**2 - (self.gamma / 2)**2)
        Gamma = self.gamma / 2

        ck_real = ([
            (self.lam**2 / (4 * Om))
            * (1 / np.tanh(beta * (Om + 1j * Gamma) / 2)),
            (self.lam**2 / (4 * Om))
            * (1 / np.tanh(beta * (Om - 1j * Gamma) / 2)),
        ])

        ck_real.extend([
            (-2 * self.lam**2 * self.gamma / beta) * (2 * np.pi * k / beta)
            / (
                ((Om + 1j * Gamma)**2 + (2 * np.pi * k / beta)**2)
                * ((Om - 1j * Gamma)**2 + (2 * np.pi * k / beta)**2)
            )
            for k in range(1, Nk + 1)
        ])

        vk_real = [-1j * Om + Gamma, 1j * Om + Gamma]
        vk_real.extend([
            2 * np.pi * k * self.T
            for k in range(1, Nk + 1)
        ])

        ck_imag = [
            1j * self.lam**2 / (4 * Om),
            -1j * self.lam**2 / (4 * Om),
        ]

        vk_imag = [-1j * Om + Gamma, 1j * Om + Gamma]

        return ck_real, vk_real, ck_imag, vk_imag


class CFExponent:
    """
    Represents a single exponent (naively, an excitation mode) within an
    exponential decomposition of the correlation function of a environment.

    Parameters
    ----------
    type : {"R", "I", "RI", "+", "-"} or one of `CFExponent.types`
        The type of exponent.

        "R" and "I" are bosonic exponents that appear in the real and
        imaginary parts of the correlation expansion, respectively.

        "RI" is a combined bosonic exponent that appears in both the real
        and imaginary parts of the correlation expansion. The combined exponent
        has a single ``vk``. The ``ck`` is the coefficient in the real
        expansion and ``ck2`` is the coefficient in the imaginary expansion.

        "+" and "-" are fermionic exponents. These fermionic exponents must
        specify ``sigma_bar_k_offset`` which specifies the amount to add to
        ``k`` (the exponent index within the environment of this exponent) to
        determine the ``k`` of the corresponding exponent with the opposite
        sign (i.e. "-" or "+").

    ck : complex
        The coefficient of the excitation term.

    vk : complex
        The frequency of the exponent of the excitation term.

    ck2 : optional, complex
        For exponents of type "RI" this is the coefficient of the term in the
        imaginary expansion (and ``ck`` is the coefficient in the real
        expansion).

    sigma_bar_k_offset : optional, int
        For exponents of type "+" this gives the offset (within the list of
        exponents within the environment) of the corresponding "-" type
        exponent. For exponents of type "-" it gives the offset of the
        corresponding "+" exponent.

    tag : optional, str, tuple or any other object
        A label for the exponent (often the name of the environment). It
        defaults to None.

    Attributes
    ----------
    fermionic : bool
        True if the type of the exponent is a Fermionic type (i.e. either
        "+" or "-") and False otherwise.

    coefficient : complex
        The coefficient of this excitation term in the total correlation
        function (including real and imaginary part).

    exponent : complex
        The frequency of the exponent of the excitation term. (Alias for `vk`.)

    All of the parameters are also available as attributes.
    """
    types = enum.Enum("ExponentType", ["R", "I", "RI", "+", "-"])

    def _check_ck2(self, type, ck2):
        if type == self.types["RI"]:
            if ck2 is None:
                raise ValueError("RI exponents require ck2")
        else:
            if ck2 is not None:
                raise ValueError(
                    "Second co-efficient (ck2) should only be specified for"
                    " RI exponents"
                )

    def _check_sigma_bar_k_offset(self, type, offset):
        if type in (self.types["+"], self.types["-"]):
            if offset is None:
                raise ValueError(
                    "+ and - type exponents require sigma_bar_k_offset"
                )
        else:
            if offset is not None:
                raise ValueError(
                    "Offset of sigma bar (sigma_bar_k_offset) should only be"
                    " specified for + and - type exponents"
                )

    def _type_is_fermionic(self, type):
        return type in (self.types["+"], self.types["-"])

    def __init__(
            self, type: str | CFExponent.ExponentType,
            ck: complex, vk: complex, ck2: complex = None,
            sigma_bar_k_offset: int = None, tag: Any = None
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self._check_ck2(type, ck2)
        self._check_sigma_bar_k_offset(type, sigma_bar_k_offset)

        self.type = type
        self.ck = ck
        self.vk = vk
        self.ck2 = ck2
        self.sigma_bar_k_offset = sigma_bar_k_offset

        self.tag = tag
        self.fermionic = self._type_is_fermionic(type)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" ck={self.ck!r} vk={self.vk!r} ck2={self.ck2!r}"
            f" sigma_bar_k_offset={self.sigma_bar_k_offset!r}"
            f" fermionic={self.fermionic!r}"
            f" tag={self.tag!r}>"
        )

    @property
    def coefficient(self) -> complex:
        coeff = 0
        if (self.type == self.types['R'] or self.type == self.types['RI']):
            coeff += self.ck
        if self.type == self.types['I']:
            coeff += 1j * self.ck
        if self.type == self.types['RI']:
            coeff += 1j * self.ck2
        return coeff

    @property
    def exponent(self) -> complex:
        return self.vk

    def _can_combine(self, other, rtol, atol):
        if type(self) is not type(other):
            return False
        if self.fermionic or other.fermionic:
            return False
        if not np.isclose(self.vk, other.vk, rtol=rtol, atol=atol):
            return False
        return True

    def _combine(self, other, **init_kwargs):
        # Assumes can combine was checked
        cls = type(self)

        if self.type == other.type and self.type != self.types['RI']:
            # Both R or both I
            return cls(type=self.type, ck=(self.ck + other.ck),
                       vk=self.vk, tag=self.tag, **init_kwargs)

        # Result will be RI
        real_part_coefficient = 0
        imag_part_coefficient = 0
        if self.type == self.types['RI'] or self.type == self.types['R']:
            real_part_coefficient += self.ck
        if other.type == self.types['RI'] or other.type == self.types['R']:
            real_part_coefficient += other.ck
        if self.type == self.types['I']:
            imag_part_coefficient += self.ck
        if other.type == self.types['I']:
            imag_part_coefficient += other.ck
        if self.type == self.types['RI']:
            imag_part_coefficient += self.ck2
        if other.type == self.types['RI']:
            imag_part_coefficient += other.ck2

        return cls(type=self.types['RI'], ck=real_part_coefficient, vk=self.vk,
                   ck2=imag_part_coefficient, tag=self.tag, **init_kwargs)


class ExponentialBosonicEnvironment(BosonicEnvironment):
    """
    Bosonic environment that is specified through an exponential decomposition
    of its correlation function. The list of coefficients and exponents in
    the decomposition may either be passed through the four lists `ck_real`,
    `vk_real`, `ck_imag`, `vk_imag`, or as a list of bosonic
    :class:`CFExponent` objects.

    Parameters
    ----------
    ck_real : list of complex
        The coefficients of the expansion terms for the real part of the
        correlation function. The corresponding frequencies are passed as
        vk_real.

    vk_real : list of complex
        The frequencies (exponents) of the expansion terms for the real part of
        the correlation function. The corresponding ceofficients are passed as
        ck_real.

    ck_imag : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vk_imag.

    vk_imag : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding ceofficients are
        passed as ck_imag.

    exponents : list of :class:`CFExponent`
        The expansion coefficients and exponents of both the real and the
        imaginary parts of the correlation function as :class:`CFExponent`
        objects.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`combine` for details.

    T: optional, float
        The temperature of the bath.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    _make_exponent = CFExponent

    def _check_cks_and_vks(self, ck_real, vk_real, ck_imag, vk_imag):
        # all None: returns False
        # all provided and lengths match: returns True
        # otherwise: raises ValueError
        lists = [ck_real, vk_real, ck_imag, vk_imag]
        if all(x is None for x in lists):
            return False
        if any(x is None for x in lists):
            raise ValueError(
                "If any of the exponent lists ck_real, vk_real, ck_imag, "
                "vk_imag is provided, all must be provided."
            )
        if len(ck_real) != len(vk_real) or len(ck_imag) != len(vk_imag):
            raise ValueError(
                "The exponent lists ck_real and vk_real, and ck_imag and "
                "vk_imag must be the same length."
            )
        return True

    def __init__(
        self,
        ck_real: ArrayLike = None, vk_real: ArrayLike = None,
        ck_imag: ArrayLike = None, vk_imag: ArrayLike = None,
        *,
        exponents: Sequence[CFExponent] = None,
        combine: bool = True, T: float = None, tag: Any = None
    ):
        super().__init__(T, tag)

        lists_provided = self._check_cks_and_vks(
            ck_real, vk_real, ck_imag, vk_imag)
        if exponents is None and not lists_provided:
            raise ValueError(
                "Either the parameter `exponents` or the parameters "
                "`ck_real`, `vk_real`, `ck_imag`, `vk_imag` must be provided."
            )
        if exponents is not None and any(exp.fermionic for exp in exponents):
            raise ValueError(
                "Fermionic exponent passed to exponential bosonic environment."
            )

        exponents = exponents or []
        if lists_provided:
            exponents.extend(self._make_exponent("R", ck, vk, tag=tag)
                             for ck, vk in zip(ck_real, vk_real))
            exponents.extend(self._make_exponent("I", ck, vk, tag=tag)
                             for ck, vk in zip(ck_imag, vk_imag))

        if combine:
            exponents = self.combine(exponents)
        self.exponents = exponents

    @classmethod
    def combine(
        cls, exponents: Sequence[CFExponent],
        rtol: float = 1e-5, atol: float = 1e-7
    ) -> Sequence[CFExponent]:
        """
        Group bosonic exponents with the same frequency and return a
        single exponent for each frequency present.

        Parameters
        ----------
        exponents : list of :class:`CFExponent`
            The list of exponents to combine.

        rtol : float, default 1e-5
            The relative tolerance to use to when comparing frequencies.

        atol : float, default 1e-7
            The absolute tolerance to use to when comparing frequencies.

        Returns
        -------
        list of :class:`CFExponent`
            The new reduced list of exponents.
        """
        remaining = exponents[:]
        new_exponents = []

        while remaining:
            new_exponent = remaining.pop(0)
            for other_exp in remaining[:]:
                if new_exponent._can_combine(other_exp, rtol, atol):
                    new_exponent = new_exponent._combine(other_exp)
                    remaining.remove(other_exp)
            new_exponents.append(new_exponent)

        return new_exponents

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Computes the correlation function represented by this exponential
        decomposition.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """

        t = np.array(t, dtype=float)
        corr = np.zeros_like(t, dtype=complex)

        for exp in self.exponents:
            corr += exp.coefficient * np.exp(-exp.exponent * np.abs(t))
        corr[t < 0] = np.conj(corr[t < 0])

        return corr

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        w = np.array(w, dtype=float)
        S = np.zeros_like(w)

        for exp in self.exponents:
            S += 2 * np.real(
                exp.coefficient / (exp.exponent - 1j * w)
            )

        return S

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the spectral density corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        return self._sd_from_ps(w)

    def to_bath(self, Q, dim=None):
        """
        It generates a BosonicBath from a BosonicEnviroment

        Parameters
        ----------
        Q : Qobj
            The Bath-system coupling operator
        dim : int or None
            The dimension (i.e. maximum number of excitations for this exponent).
            Usually ``2`` for fermionic exponents or ``None`` (i.e. unlimited) for
            bosonic exponents. For a better description see `BosonicBath`

        Returns
        -------
        result: BosonicBath
            A bosonic bath object that describes the BosonicEnvironment
        """
        bath_exponents = []
        maps = {}
        for exponent in self.exponents:
            new_exponent = BathExponent(
                exponent.type.name, dim, Q, exponent.ck, exponent.vk,
                exponent.ck2, exponent.sigma_bar_k_offset, self.tag
            )
            bath_exponents.append(new_exponent)

        result = BosonicBath(Q, [], [], [], [], tag=self.tag)
        result.exponents = bath_exponents
        return result


class OhmicEnvironment(BosonicEnvironment):
    r"""
    Describes Ohmic environments as well as sub- or super-Ohmic environments
    (depending on the choice of the parameter `s`). The spectral density is

    .. math::
        J(\omega)
        = \alpha \frac{\omega^s}{\omega_c^{1-s}} e^{-\omega / \omega_c} .

    This class requires the `mpmath` module to be installed.

    Parameters
    ----------
    T : float
        Temperature of the bath.

    alpha : float
        Coupling strength.

    wc : float
        Cutoff parameter.

    s : float
        Power of omega in the spectral density.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, alpha: float, wc: float, s: float, *, tag: Any = None
    ):
        super().__init__(T, tag)

        self.alpha = alpha
        self.wc = wc
        self.s = s

        if _mpmath_available is False:
            warnings.warn(
                "The mpmath module is required for some operations on "
                "Ohmic environments, but it is not installed.")

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        r"""
        Calculates the spectral density of the Ohmic environment.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.array(w, dtype=float)
        result = np.zeros_like(w)

        positive_mask = (w > 0)
        w_mask = w[positive_mask]
        result[positive_mask] = (
            self.alpha * w_mask ** self.s
            / (self.wc ** (1 - self.s))
            * np.exp(-np.abs(w_mask) / self.wc)
        )

        return result

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum of the Ohmic environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """
        if self.s > 1:
            sd_derivative = 0
        elif self.s == 1:
            sd_derivative = self.alpha
        else:
            sd_derivative = np.inf
        return self._ps_from_sd(w, None, sd_derivative)

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        r"""
        Calculates the correlation function of an Ohmic bath using the formula

        .. math::
            C(t)= \frac{1}{\pi} \alpha w_{c}^{1-s} \beta^{-(s+1)} \Gamma(s+1)
            \left[ \zeta\left(s+1,\frac{1+\beta w_{c} -i w_{c} t}{\beta w_{c}}
            \right) +\zeta\left(s+1,\frac{1+ i w_{c} t}{\beta w_{c}}\right)
            \right] ,

        where :math:`\Gamma` is the gamma function, and :math:`\zeta` the
        Riemann zeta function.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """
        t = np.array(t, dtype=float)
        t_was_array = t.ndim > 0
        if not t_was_array:
            t = np.array([t], dtype=float)

        if self.T != 0:
            corr = (self.alpha * self.wc ** (1 - self.s) / np.pi
                    * mp.gamma(self.s + 1) * self.T ** (self.s + 1))
            z1_u = ((1 + self.wc / self.T - 1j * self.wc * t)
                    / (self.wc / self.T))
            z2_u = (1 + 1j * self.wc * t) / (self.wc / self.T)
            result = np.array(
                [corr * (mp.zeta(self.s + 1, u1) + mp.zeta(self.s + 1, u2))
                 for u1, u2 in zip(z1_u, z2_u)],
                dtype=np.cdouble
            )
        else:
            corr = (self.alpha * self.wc ** (self.s+1) / np.pi
                    * mp.gamma(self.s + 1)
                    * (1 + 1j * self.wc * t) ** (-(self.s + 1)))
            result = np.array(corr, dtype=np.cdouble)

        if t_was_array:
            return result
        return result[0]


# --- utility functions ---

def _real_interpolation(fun, xlist, name):
    if callable(fun):
        return fun
    else:
        if xlist is None or len(xlist) != len(fun):
            raise ValueError("A list of x-values with the same length must be "
                             f"provided for the discretized function ({name})")
        return CubicSpline(xlist, fun)


def _complex_interpolation(fun, xlist, name):
    if callable(fun):
        return fun
    else:
        real_interp = _real_interpolation(np.real(fun), xlist, name)
        imag_interp = _real_interpolation(np.imag(fun), xlist, name)
        return lambda x: real_interp(x) + 1j * imag_interp(x)


def _fft(f, wMax, tMax):
    r"""
    Calculates the Fast Fourier transform of the given function. We calculate
    Fourier transformations via FFT because numerical integration is often
    noisy in the scenarios we are interested in.

    Given a (mathematical) function `f(t)`, this function approximates its
    Fourier transform

    .. math::
        g(\omega) = \int_{-\infty}^\infty dt\, e^{-i\omega t}\, f(t) .

    The function f is sampled on the interval `[-tMax, tMax]`. The sampling
    discretization is chosen as `dt = pi / (2*wMax)` (Shannon-Nyquist + some
    leeway). However, `dt` is always chosen small enough to have at least 250
    samples on the interval `[-tMax, tMax]`.

    Parameters
    ----------
    wMax: float
        Maximum frequency of interest
    tMax: float
        Support of the function f (i.e., f(t) is essentially zero for
        `|t| > tMax`).

    Returns
    -------
    The fourier transform of the provided function as an interpolated function.
    """
    # Code adapted from https://stackoverflow.com/a/24077914

    numSamples = int(
        max(250, np.ceil(4 * tMax * wMax / np.pi + 1))
    )
    t, dt = np.linspace(-tMax, tMax, numSamples, retstep=True)
    f_values = f(t)

    # Compute Fourier transform by numpy's FFT function
    g = np.fft.fft(f_values)
    # frequency normalization factor is 2 * np.pi / dt
    w = np.fft.fftfreq(numSamples) * 2 * np.pi / dt
    # In order to get a discretisation of the continuous Fourier transform
    # we need to multiply g by a phase factor
    g *= dt * np.exp(1j * w * tMax)

    return _complex_interpolation(
        np.fft.fftshift(g), np.fft.fftshift(w), 'FFT'
    )


def _cf_real_fit_model(tlist, a, b, c, d=0):
    return np.real((a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist)))


def _cf_imag_fit_model(tlist, a, b, c, d=0):
    return np.imag((a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist)))


def _default_guess_cfreal(tlist, clist, full_ansatz):
    corr_abs = np.abs(clist)
    corr_max = np.max(corr_abs)

    # Checks if constant array, and assigns zero
    if (clist == clist[0]).all():
        if full_ansatz:
            return [[0] * 4]*3
        return [[0] * 3]*3

    if full_ansatz:
        lower = [-100 * corr_max, -np.inf, -np.inf, -100 * corr_max]
        guess = [corr_max, -100*corr_max, 0, 0]
        upper = [100*corr_max, 0, np.inf, 100*corr_max]
    else:
        lower = [-20 * corr_max, -np.inf, -np.inf]
        guess = [corr_max, 0, 0]
        upper = [20 * corr_max, 0.1, np.inf]

    return guess, lower, upper


def _default_guess_cfimag(clist, full_ansatz):
    corr_max = np.max(np.abs(clist))

    # Checks if constant array, and assigns zero
    if (clist == clist[0]).all():
        if full_ansatz:
            return [[0] * 4]*3
        return [[0] * 3]*3

    if full_ansatz:
        lower = [-100 * corr_max, -np.inf, -np.inf, -100 * corr_max]
        guess = [0, -10 * corr_max, 0, 0]
        upper = [100 * corr_max, 0, np.inf, 100 * corr_max]
    else:
        lower = [-20 * corr_max, -np.inf, -np.inf]
        guess = [-corr_max, -10 * corr_max, 1]
        upper = [10 * corr_max, 0, np.inf]

    return guess, lower, upper


def _sd_fit_model(wlist, a, b, c):
    return (
        2 * a * b * wlist / ((wlist + c)**2 + b**2) / ((wlist - c)**2 + b**2)
    )


def _default_guess_sd(wlist, jlist):
    sd_abs = np.abs(jlist)
    sd_max = np.max(sd_abs)
    wc = wlist[np.argmax(sd_abs)]

    if sd_max == 0:
        return [0] * 3

    lower = [-100 * sd_max, 0.1 * wc, 0.1 * wc]
    guess = [sd_max, wc, wc]
    upper = [100 * sd_max, 100 * wc, 100 * wc]

    return guess, lower, upper


def _fit_summary(time, rmse, N, label, params,
                 columns=['lam', 'gamma', 'w0']):
    # Generates summary of fit by nonlinear least squares
    if len(columns) == 3:
        summary = (f"Result of fitting {label} "
                   f"with {N} terms: \n \n {'Parameters': <10}|"
                   f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: >5} \n ")
        for k in range(N):
            summary += (
                f"{k+1: <10}|{params[k][0]: ^10.2e}|{params[k][1]:^10.2e}|"
                f"{params[k][2]:>5.2e}\n ")
    elif len(columns) == 4:
        summary = (
            f"Result of fitting {label} "
            f"with {N} terms: \n \n {'Parameters': <10}|"
            f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: ^10}"
            f"|{columns[3]: >5} \n ")
        for k in range(N):
            summary += (
                f"{k+1: <10}|{params[k][0]: ^10.2e}|{params[k][1]:^10.2e}"
                f"|{params[k][2]:^10.2e}|{params[k][3]:>5.2e}\n ")
    else:
        raise ValueError("Unsupported number of columns")
    summary += (f"\nA normalized RMSE of {rmse: .2e}"
                f" was obtained for the {label}.\n")
    summary += f"The current fit took {time: 2f} seconds."
    return summary


def _cf_fit_summary(
    params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
    rmse_real, rmse_imag, n=3
):
    # Generate nicely formatted summary with two columns for CF fit
    columns = ["a", "b", "c"]
    if n == 4:
        columns.append("d")
    summary_real = _fit_summary(
        fit_time_real, rmse_real, Nr,
        "the real part of\nthe correlation function",
        params_real, columns=columns
    )
    summary_imag = _fit_summary(
        fit_time_imag, rmse_imag, Ni,
        "the imaginary part\nof the correlation function",
        params_imag, columns=columns
    )

    full_summary = "Correlation function fit:\n\n"
    lines_real = summary_real.splitlines()
    lines_imag = summary_imag.splitlines()
    max_lines = max(len(lines_real), len(lines_imag))
    # Fill the shorter string with blank lines
    lines_real = (
        lines_real[:-1]
        + (max_lines - len(lines_real)) * [""] + [lines_real[-1]]
    )
    lines_imag = (
        lines_imag[:-1]
        + (max_lines - len(lines_imag)) * [""] + [lines_imag[-1]]
    )
    # Find the maximum line length in each column
    max_length1 = max(len(line) for line in lines_real)
    max_length2 = max(len(line) for line in lines_imag)

    # Print the strings side by side with a vertical bar separator
    for line1, line2 in zip(lines_real, lines_imag):
        formatted_line1 = f"{line1:<{max_length1}} |"
        formatted_line2 = f"{line2:<{max_length2}}"
        full_summary += formatted_line1 + formatted_line2 + "\n"
    return full_summary


# -----------------------------------------------------------------------------
# Fitting utilities
#

def iterated_fit(
    fun: Callable[..., complex], num_params: int,
    xdata: ArrayLike, ydata: ArrayLike,
    target_rmse: float = 1e-5,
    guess: ArrayLike | Callable[[int], ArrayLike] = None,
    Nmin: int = 1, Nmax: int = 10,
    lower: ArrayLike = None, upper: ArrayLike = None,
) -> tuple[float, ArrayLike]:
    r"""
    Iteratively tries to fit the given data with a model of the form

    .. math::
        y = \sum_{k=1}^N f(x; p_{k,1}, \dots, p_{k,n})

    where `f` is a model function depending on `n` parameters, and the number
    `N` of terms is increases until the normalized rmse (root mean square
    error) falls below the target value.

    Parameters
    ----------
    fun : callable
        The model function. Its first argument is the array `xdata`, its other
        arguments are the fitting parameters.
    num_params : int
        The number of fitting parameters per term (`n` in the equation above).
        The function `fun` must take `num_params+1` arguments.
    xdata : array_like
        The independent variable.
    ydata : array_like
        The dependent data.
    target_rmse : optional, float
        Desired normalized root mean squared error (default `1e-5`).
    guess : optional, array_like or callable
        This can be either a list of length `n`, with the i-th entry being the
        guess for the parameter :math:`p_{k,i}` (for all terms :math:`k`), or a
        function that provides different initial guesses for each term.
        Specifically, given a number `N` of terms, the function returns an
        array `[[p11, ..., p1n], [p21, ..., p2n], ..., [pN1, ..., pNn]]` of
        initial guesses.
    Nmin : optional, int
        The minimum number of terms to be used for the fit (default 1).
    Nmax : optional, int
        The maximum number of terms to be used for the fit (default 10).
        If the number `Nmax` of terms is reached, the function returns even if
        the target rmse has not been reached yet.
    lower : optional, list of length `num_params`
        Lower bounds on the parameters for the fit.
    upper : optional, list of length `num_params`
        Upper bounds on the parameters for the fit.

    Returns
    -------
    rmse : float
        The normalized mean squared error of the fit
    params : array_like
        The model parameters in the form
        `[[p11, ..., p1n], [p21, ..., p2n], ..., [pN1, ..., pNn]]`.
    """

    # Check if array is constant
    # if (ydata == ydata[0]).all():
    #     if num_params == 3:
    #         return [ydata[0]], [0], [0]
    #     else:
    #         return [ydata[0]], [0], [0], [0]
    if len(xdata) != len(ydata):
        raise ValueError(
            "The shape of the provided fit data is not consistent")

    if lower is None:
        lower = np.full(num_params, -np.inf)
    if upper is None:
        upper = np.full(num_params, np.inf)
    if not (len(lower) == num_params and len(upper) == num_params):
        raise ValueError(
            "The shape of the provided fit bounds is not consistent")

    N = Nmin
    rmse1 = np.inf

    while rmse1 > target_rmse and N <= Nmax:
        if guess is None:
            guesses = np.ones((N, num_params), dtype=float)
        elif callable(guess):
            guesses = np.array(guess(N))
            if guesses.shape != (N, num_params):
                raise ValueError(
                    "The shape of the provided fit guesses is not consistent")
        else:
            guesses = np.tile(guess, (N, 1))

        lower_repeat = np.tile(lower, N)
        upper_repeat = np.tile(upper, N)
        rmse1, params = _fit(fun, num_params, xdata, ydata,
                             guesses, lower_repeat, upper_repeat)
        N += 1

    return rmse1, params


def _pack(params):
    # Pack parameter lists for fitting.
    # Input: array of parameters like `[[p11, ..., p1n], ..., [pN1, ..., pNn]]`
    # Output: packed parameters like `[p11, ..., p1n, p21, ..., p2n, ...]`
    return params.ravel()  # like flatten, but doesn't copy data


def _unpack(params, num_params):
    # Inverse of _pack, `num_params` is "n"
    N = len(params) // num_params
    return np.reshape(params, (N, num_params))


def _evaluate(fun, x, params):
    result = 0
    for term_params in params:
        result += fun(x, *term_params)
    return result


def _rmse(fun, xdata, ydata, params):
    """
    The normalized root mean squared error for the fit with the given
    parameters. (The closer to zero = the better the fit.)
    """
    yhat = _evaluate(fun, xdata, params)
    if (np.max(ydata) - np.min(ydata)) == 0.0:
        return 0
    return (
        np.sqrt(np.mean((yhat - ydata) ** 2) / len(ydata))
        / (np.max(ydata) - np.min(ydata))
    )


def _fit(fun, num_params, xdata, ydata, guesses, lower, upper):
    # fun: model function
    # num_params: number of parameters in fun
    # xdata, ydata: data to be fit
    # N: number of terms
    # guesses: initial guesses [[p11, ..., p1n],..., [pN1, ..., pNn]]
    # lower, upper: parameter bounds
    if (upper <= lower).all():
        return _rmse(fun, xdata, ydata, guesses), guesses

    packed_params, _ = curve_fit(
        lambda x, *packed_params: _evaluate(
            fun, x, _unpack(packed_params, num_params)
        ),
        xdata, ydata, p0=_pack(guesses), bounds=(lower, upper),
        maxfev=int(1e9), method="trf"
    )
    params = _unpack(packed_params, num_params)
    rmse = _rmse(fun, xdata, ydata, params)
    return rmse, params


def n_thermal(w, w_th):
    """
    Return the number of photons in thermal equilibrium for an harmonic
    oscillator mode with frequency 'w', at the temperature described by
    'w_th' where :math:`\\omega_{\\rm th} = k_BT/\\hbar`.

    Parameters
    ----------

    w : float or ndarray
        Frequency of the oscillator.

    w_th : float
        The temperature in units of frequency (or the same units as `w`).


    Returns
    -------

    n_avg : float or array

        Return the number of average photons in thermal equilibrium for a
        an oscillator with the given frequency and temperature.


    """

    if w_th <= 0:
        return np.zeros_like(w)

    w = np.array(w, dtype=float)
    result = np.zeros_like(w)
    non_zero = w != 0
    result[non_zero] = 1 / (np.exp(w[non_zero] / w_th) - 1)
    return result
