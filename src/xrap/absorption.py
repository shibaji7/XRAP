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

Low-frequency saturation
------------------------
``(30/f)**1.5`` diverges toward 0 Hz (22.6x at 3.75 MHz, 68x at 1.8 MHz). XRAP
therefore saturates it below ``freq_saturation_mhz`` (default 5 MHz; continuous,
power law unchanged above, bounded by ``2 (30/5)**1.5 = 29.4`` as f -> 0); see
:func:`scale_frequency`. DRAP2 keeps its published, unsaturated form unless you
opt in. Pass ``freq_saturation_mhz=None`` to disable, or a float to move the
corner.

Flux scale
----------
``flux`` must be on the **NOAA flare-class scale** (``X9.3 = 9.3e-4 W m-2``),
which is what both models are calibrated on. Science-quality GOES L2 data are
~1/0.7 larger in the long band; :func:`xrap.fetch_goes_xrs` converts them
(``flare_scale=True``, default). If you pass your own science-quality flux,
multiply it by 0.7 first.

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
    "peak_scale",
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
# Peak scaling (max_db)
# --------------------------------------------------------------------------- #
def peak_scale(values, max_db: float):
    """Scale a whole array by one factor so that its peak equals ``max_db``.

    Down only: if the array's maximum is already ``<= max_db`` (or is not a
    positive finite number) it is returned unchanged with factor 1. Otherwise
    every element is multiplied by ``max_db / max(values)``, so the shape of a
    time series -- the *relative* variation -- is preserved and its peak lands
    exactly on ``max_db``.

    Note that the factor depends on the whole array you pass (e.g. the length of
    the time window), and that it rescales quiet values too; two series scaled
    separately are no longer comparable with each other.

    Returns
    -------
    (scaled, factor) : (numpy.ndarray, float)
    """
    v = np.asarray(values, dtype=float)
    peak = np.nanmax(v) if v.size and not np.all(np.isnan(v)) else np.nan
    if not np.isfinite(peak) or peak <= 0.0 or peak <= max_db:
        return v, 1.0
    factor = float(max_db) / float(peak)
    # pin the peak element(s) to exactly max_db (avoids 99.99999999999999)
    return np.where(v == peak, float(max_db), v * factor), factor


def _apply_max_db(values, max_db: float | None, mode: str) -> np.ndarray:
    """Apply the ``max_db`` limit: ``"scale"`` (peak scaling) or ``"clip"`` (hard cut)."""
    if max_db is None:
        return values
    if not max_db > 0.0:
        raise ValueError(f"max_db must be positive, got {max_db!r}")
    if mode == "scale":
        return peak_scale(values, max_db)[0]
    if mode == "clip":
        return np.minimum(values, max_db)
    raise ValueError(f"max_db_mode must be 'scale' or 'clip', got {mode!r}")


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
    max_db: float | None = None,
    max_db_mode: str = "scale",
    freq_saturation_mhz: float | str | None = "default",
) -> np.ndarray:
    """Solar-flare D-region absorption in dB.

    Parameters
    ----------
    freq_mhz : float or array-like
        HF wave frequency (MHz).
    sza_deg : float or array-like
        Solar zenith angle (degrees).
    xray_wm2 : float or array-like
        GOES 0.1-0.8 nm long-band flux (W m-2) on the NOAA flare-class scale
        (``X9.3 = 9.3e-4``; see the module docstring).
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
    max_db : float, optional
        Limit on the result (dB). Neither model saturates, so low frequencies /
        strong flares can reach hundreds of dB. With ``max_db_mode="scale"``
        (default) the **whole returned array is multiplied by one factor so its
        peak equals** ``max_db`` (only if the peak exceeds it; see
        :func:`peak_scale`) -- the time-series shape is kept, no flat top. A
        scalar result above ``max_db`` simply becomes ``max_db``. ``None``
        (default) leaves the raw value.
    max_db_mode : {"scale", "clip"}
        ``"scale"`` (default): peak scaling as above. ``"clip"``: hard cut at
        ``max_db`` (``result >= max_db`` then means "blackout").
    freq_saturation_mhz : float, None or "default"
        Low-frequency saturation corner (MHz; see :func:`scale_frequency`): the
        ``f**-n`` law is replaced below this frequency by a bounded one, so the
        absorption stops diverging toward 0 Hz. ``"default"`` (default) uses the
        model's own setting -- 5 MHz for ``"xrap"``, off for ``"drap2"`` (faithful
        to its published form); a float overrides it; ``None`` disables it.

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

    gz = dict(grazing=grazing, alt_km=alt_km, H_km=H_km,
              f_sat=_resolve_saturation(freq_saturation_mhz, c))
    if model == "xrap":
        L = _absorption_xrap(f, chi, phi, c, **gz)
    else:
        L = _absorption_drap2(f, chi, phi, c, **gz)

    if path == "total":
        L = 2.0 * L
    if clip_negative:
        L = np.maximum(L, 0.0)
    return _apply_max_db(L, max_db, max_db_mode)


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
# Frequency law (shared), with optional low-frequency saturation
# --------------------------------------------------------------------------- #
def _frequency_law(f_mhz, n: float, f_sat_mhz: float | None):
    """The ``f**-n`` frequency dependence, optionally saturated below ``f_sat_mhz``.

    ``f_sat_mhz is None``  ->  ``f**-n`` (unbounded as f -> 0).
    otherwise, ``f**-n`` for ``f >= f_sat`` and ``2 / (f**n + f_sat**n)`` below it:
    continuous at ``f_sat`` (both equal ``f_sat**-n``), bounded by
    ``2 f_sat**-n`` as f -> 0. The power law above ``f_sat`` is left exactly
    unchanged.
    """
    f = np.asarray(f_mhz, dtype=float)
    pure = f ** (-n)
    if f_sat_mhz is None:
        return pure
    return np.where(f >= f_sat_mhz, pure, 2.0 / (f**n + f_sat_mhz**n))


def _resolve_saturation(freq_saturation_mhz, coeffs: dict):
    """``"default"`` -> the model's own ``freq_saturation_mhz`` (None if it has none)."""
    if isinstance(freq_saturation_mhz, str):
        if freq_saturation_mhz != "default":
            raise ValueError(
                f"freq_saturation_mhz must be a float, None or 'default', got {freq_saturation_mhz!r}"
            )
        return coeffs.get("freq_saturation_mhz")
    return freq_saturation_mhz


# --------------------------------------------------------------------------- #
# Model implementations (return one-way dB)
# --------------------------------------------------------------------------- #
def _absorption_xrap(f, chi, phi, c, *, grazing, alt_km, H_km, f_sat) -> np.ndarray:
    a_f0 = reference_absorption(chi, phi, coeffs=c, grazing=grazing,
                               alt_km=alt_km, H_km=H_km)
    return scale_frequency(a_f0, c["f0_mhz"], f, n=c["freq_exponent"], f_sat_mhz=f_sat)


def _absorption_drap2(f, chi, phi, c, *, grazing, alt_km, H_km, f_sat) -> np.ndarray:
    haf = highest_affected_frequency(
        phi, chi, coeffs=c, grazing=grazing, alt_km=alt_km, H_km=H_km
    )
    n = c["freq_exponent"]
    with np.errstate(divide="ignore", invalid="ignore"):
        total = c["haf_threshold_db"] * haf**n * _frequency_law(f, n, f_sat)  # = (HAF/f)^n
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
def scale_frequency(
    absorption_db, f0_mhz, f_mhz, *, n: float | None = None, f_sat_mhz: float | None = None
):
    """Scale an absorption value from one HF frequency to another.

        A(f) = (f0 / f) ** n * A(f0)     [dB]

    With ``f_sat_mhz`` the law saturates below that frequency (continuous, and
    unchanged above it)::

        A(f) = 2 f0**n / (f**n + f_sat**n) * A(f0)       for f < f_sat

    which is bounded by ``2 (f0 / f_sat)**n * A(f0)`` as ``f -> 0`` instead of
    diverging. Physically, D-region collisions dominate at low frequency and
    absorption stops depending on frequency.

    Parameters
    ----------
    absorption_db : float or array-like
        Absorption ``A(f0)`` at the reference frequency (dB).
    f0_mhz, f_mhz : float or array-like
        Reference and target frequencies (MHz).
    n : float, optional
        Exponent. Defaults to 1.5 (Hargreaves; used by both XRAP and DRAP2);
        pass another value to override.
    f_sat_mhz : float, optional
        Saturation corner (MHz). ``None`` (default here) keeps the pure power law.

    Returns
    -------
    numpy.ndarray
        ``A(f)`` at the target frequency (dB), broadcast over the inputs.
    """
    exponent = 1.5 if n is None else n
    a0 = np.asarray(absorption_db, dtype=float)
    f0 = np.asarray(f0_mhz, dtype=float)
    return f0**exponent * _frequency_law(f_mhz, exponent, f_sat_mhz) * a0


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
