"""HF absorption models for solar-flare shortwave fadeout.

Two models, selected with ``model=``:

``"xrap"`` (default) -- Fiori, Chakraborty & Nikitina (2022)
    Data-optimized simple model, fit to the NRCan 30 MHz riometer network::

        A(f0) = C * flux^p_I * g(chi)^p_c        [dB, one-way, f0 = 30 MHz]
        A(f)  = (f0 / f)^n * A(f0)                A >= 0

    with ``C = 12080``, ``p_I = p_c = 1``, ``n = 1.5`` (Hargreaves; Parthasarathy
    et al. 1963). ``flux`` is the GOES 0.1-0.8 nm irradiance in W m-2. The
    exponents are exposed so the paper's optimized coefficient sets drop in.
    doi:10.1016/j.jastp.2022.105843

``"drap2"`` -- NOAA Global D-Region Absorption Prediction v2, X-ray term::

        HAF0      = a * log10(flux) + b           [MHz]   (a = 10, b = 65)
        HAF(chi)  = HAF0 * (cos chi)^p            0 for chi >= 90 deg   (p = 0.75)
        A(f,chi)  = 0.5 * A_ref * (HAF(chi) / f)^n   [dB, one-way]   (n = 1.5)

    https://www.spaceweather.gov/content/global-d-region-absorption-prediction-documentation

Both return **one-way** vertical absorption; ``path="total"`` doubles it.

Zenith-angle handling
---------------------
``g(chi)`` / the ``(cos chi)`` taper is either

* hard (``grazing=False``) -- ``max(cos chi, 0)``, zero at ``chi = 90 deg``; or
* grazing (``grazing=True``) -- ``1 / Ch(X, chi)`` (Chapman grazing-incidence
  function, :func:`xrap.sza.chapman_function`), a small positive value through
  ``chi = 90 deg`` that decays smoothly to zero at the terminator
  (:func:`xrap.sza.terminator_sza`, ~99.6 deg at 90 km) and equals ``cos chi``
  for ``chi`` below ~80 deg.

``grazing`` defaults to **True for ``"xrap"``** (``cos chi`` there is the
overhead ionization-rate factor, for which the Chapman function is the correct
grazing generalization) and **False for ``"drap2"``** (faithful to the
published spec).

The SEP / polar-cap term is out of scope -- see :func:`polar_cap_absorption`.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files

import numpy as np

__all__ = [
    "absorption",
    "reference_absorption",
    "highest_affected_frequency",
    "scale_frequency",
    "zenith_factor",
    "polar_cap_absorption",
    "load_coefficients",
]

MODELS = ("xrap", "drap2")


@cache
def _raw_coefficients(model: str) -> tuple:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    text = files("xrap.data").joinpath(f"{model}.json").read_text()
    return tuple((k, v) for k, v in json.loads(text).items() if not k.startswith("_"))


def load_coefficients(model: str = "xrap") -> dict:
    """Return the bundled coefficient set for ``model`` (a fresh dict each call)."""
    return dict(_raw_coefficients(model))


# --------------------------------------------------------------------------- #
# Shared zenith-angle factor
# --------------------------------------------------------------------------- #
def zenith_factor(
    sza_deg,
    *,
    exponent: float = 1.0,
    grazing: bool = False,
    alt_km: float = 90.0,
    H_km: float = 7.0,
) -> np.ndarray:
    """Zenith-angle factor raised to ``exponent``.

    ``grazing=False`` -> ``max(cos chi, 0) ** exponent`` (0 for ``chi >= 90``).
    ``grazing=True``  -> ``(1 / Ch(X, chi)) ** exponent`` where the layer at
    ``alt_km`` is sunlit, else 0.
    """
    chi = np.asarray(sza_deg, dtype=float)

    if not grazing:
        cos_chi = np.cos(np.radians(chi))
        return np.where(chi < 90.0, np.maximum(cos_chi, 0.0) ** exponent, 0.0)

    from .sza import chapman_function, is_sunlit

    ch = chapman_function(chi, alt_km, H_km=H_km)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 / ch) ** exponent
    g = np.nan_to_num(g, nan=0.0, posinf=0.0)
    return np.where(is_sunlit(chi, alt_km), g, 0.0)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def absorption(
    freq_mhz,
    sza_deg,
    xray_wm2,
    *,
    model: str = "xrap",
    coeffs: dict | None = None,
    path: str = "oneway",
    grazing: bool | None = None,
    alt_km: float = 90.0,
    H_km: float = 7.0,
    clip_negative: bool = True,
) -> np.ndarray:
    """Solar-flare D-region absorption in dB.

    Parameters
    ----------
    freq_mhz : float or array-like
        HF wave frequency (MHz).
    sza_deg : float or array-like
        Solar zenith angle (degrees).
    xray_wm2 : float or array-like
        GOES 0.1-0.8 nm long-band flux (W m-2).
    model : {"xrap", "drap2"}
        Which model (see module docstring). Default ``"xrap"``.
    coeffs : dict, optional
        Coefficient override; defaults to the bundled set for ``model``.
    path : {"oneway", "total"}
        ``"oneway"`` (default) vertical absorption; ``"total"`` doubles it.
    grazing : bool, optional
        Chapman grazing correction near the terminator. ``None`` -> per-model
        default (True for ``"xrap"``, False for ``"drap2"``).
    alt_km, H_km : float
        Absorbing-layer altitude / neutral scale height (grazing only).
    clip_negative : bool
        Clip the result to >= 0 dB.

    Returns
    -------
    numpy.ndarray
        Absorption in dB, broadcast over the inputs.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if path not in ("oneway", "total"):
        raise ValueError(f"path must be 'oneway' or 'total', got {path!r}")

    c = coeffs if coeffs is not None else load_coefficients(model)
    if grazing is None:
        grazing = model == "xrap"

    f = np.asarray(freq_mhz, dtype=float)
    chi = np.asarray(sza_deg, dtype=float)
    phi = np.asarray(xray_wm2, dtype=float)

    gz = dict(grazing=grazing, alt_km=alt_km, H_km=H_km)
    if model == "xrap":
        L = _absorption_xrap(f, chi, phi, c, **gz)
    else:
        L = _absorption_drap2(f, chi, phi, c, **gz)

    if path == "total":
        L = 2.0 * L
    if clip_negative:
        L = np.maximum(L, 0.0)
    return L


def reference_absorption(
    sza_deg,
    xray_wm2,
    *,
    coeffs: dict | None = None,
    grazing: bool | None = None,
    alt_km: float = 90.0,
    H_km: float = 7.0,
) -> np.ndarray:
    """XRAP one-way absorption at the reference frequency ``f0`` (30 MHz), in dB.

    ``A(f0) = C * flux^p_I * g(chi)^p_c``. Feed this to :func:`scale_frequency`
    to reach another frequency.
    """
    c = coeffs if coeffs is not None else load_coefficients("xrap")
    if grazing is None:
        grazing = True
    phi = np.asarray(xray_wm2, dtype=float)
    flux = np.where(phi > 0.0, phi, 0.0) ** c["flux_exponent"]
    g = zenith_factor(sza_deg, exponent=c["cos_exponent"],
                      grazing=grazing, alt_km=alt_km, H_km=H_km)
    return np.maximum(c["scale"] * flux * g, 0.0)


# --------------------------------------------------------------------------- #
# Model implementations (return one-way dB)
# --------------------------------------------------------------------------- #
def _absorption_xrap(f, chi, phi, c, *, grazing, alt_km, H_km) -> np.ndarray:
    a_f0 = reference_absorption(chi, phi, coeffs=c, grazing=grazing,
                               alt_km=alt_km, H_km=H_km)
    return scale_frequency(a_f0, c["f0_mhz"], f, n=c["freq_exponent"])


def _absorption_drap2(f, chi, phi, c, *, grazing, alt_km, H_km) -> np.ndarray:
    haf = highest_affected_frequency(
        phi, chi, coeffs=c, grazing=grazing, alt_km=alt_km, H_km=H_km
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        total = c["haf_threshold_db"] * (haf / f) ** c["freq_exponent"]
    total = np.where(haf > 0.0, total, 0.0)
    return 0.5 * total  # one-way


# --------------------------------------------------------------------------- #
# DRAP2 intermediate: Highest Affected Frequency
# --------------------------------------------------------------------------- #
def highest_affected_frequency(
    xray_wm2,
    sza_deg,
    *,
    coeffs: dict | None = None,
    grazing: bool = False,
    alt_km: float = 90.0,
    H_km: float = 7.0,
) -> np.ndarray:
    """DRAP2 Highest Affected Frequency ``HAF(chi)`` in MHz.

    ``HAF0 = a*log10(flux) + b`` (clipped >= 0), tapered by
    ``(cos chi)^sza_exponent`` (or the grazing factor). 0 where the point is
    dark or the flux is below the HAF=0 level (~3.2e-7 W m-2).
    """
    c = coeffs if coeffs is not None else load_coefficients("drap2")
    phi = np.asarray(xray_wm2, dtype=float)
    safe = np.where(phi > 0.0, phi, np.nan)
    with np.errstate(invalid="ignore"):
        haf0 = c["haf_slope"] * np.log10(safe) + c["haf_intercept"]
    haf0 = np.nan_to_num(np.maximum(haf0, 0.0), nan=0.0)

    taper = zenith_factor(sza_deg, exponent=c["sza_exponent"],
                          grazing=grazing, alt_km=alt_km, H_km=H_km)
    return haf0 * taper


# --------------------------------------------------------------------------- #
# Frequency scaling
# --------------------------------------------------------------------------- #
def scale_frequency(absorption_db, f0_mhz, f_mhz, *, n: float | None = None):
    """Scale an absorption value from one HF frequency to another.

        A(f) = (f0 / f) ** n * A(f0)     [dB]

    Parameters
    ----------
    absorption_db : float or array-like
        Absorption ``A(f0)`` at the reference frequency (dB).
    f0_mhz, f_mhz : float or array-like
        Reference and target frequencies (MHz).
    n : float, optional
        Exponent. Defaults to 1.5 (Hargreaves; used by both XRAP and DRAP2);
        pass another value to override.

    Returns
    -------
    numpy.ndarray
        ``A(f)`` at the target frequency (dB), broadcast over the inputs.
    """
    exponent = 1.5 if n is None else n
    a0 = np.asarray(absorption_db, dtype=float)
    f0 = np.asarray(f0_mhz, dtype=float)
    f = np.asarray(f_mhz, dtype=float)
    return (f0 / f) ** exponent * a0


def polar_cap_absorption(*args, **kwargs):
    """DRAP2 solar-energetic-proton (polar-cap) absorption term.

    **Not implemented.** DRAP2's second component drives absorption inside the
    polar cap from the >10 MeV proton flux (GOES SGPS/EPEAD), with a
    day/night-dependent form. Out of scope for the shortwave-fadeout models
    here; add it when SEP events are needed.
    """
    raise NotImplementedError(
        "polar-cap (SEP) absorption is not implemented; XRAP covers the "
        "solar-flare (X-ray) shortwave-fadeout term only"
    )
