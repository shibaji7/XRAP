"""Oblique-path (transmitter-to-receiver) HF absorption via the secant law.

:func:`xrap.absorption.absorption` gives **vertical** D-region absorption. A
real HF circuit is oblique: the wave leaves the transmitter at some elevation
angle, is refracted back by the F layer one or more hops later, and crosses
the D region twice per hop (once going up, once coming down) -- each crossing
generally at a different location (hence a different solar zenith angle) and
at non-vertical incidence.

The classical (non-deviative-absorption) treatment -- the **secant law** /
Martyn's theorem -- says a ray hitting the D region at incidence angle ``chi_i``
(from vertical) behaves like a *vertical* ray at the equivalent frequency
``f * cos(chi_i)``::

    L_oblique(f, chi_i) = sec(chi_i) * L_vertical(f * cos(chi_i))

Both bundled models have ``L_vertical(f) ~ f**-n`` (``n = freq_exponent``,
1.5), so this collapses to a closed-form obliquity factor::

    L_oblique(f, chi_i) = sec(chi_i)**(n + 1) * L_vertical(f)      # see obliquity_factor

This module supplies the geometry (:func:`hop_crossings`) to locate every
D-region crossing of an ``n_hops``-hop path and its incidence angle, then
sums the per-crossing, obliquity-corrected, SZA-dependent vertical absorption
(:func:`oblique_absorption`).

Scope and approximations
-------------------------
* Non-deviative D-region absorption only (same regime as the vertical models);
  no F-region deviative-absorption or MUF-cusp enhancement.
* The two D-region crossings of a hop are located along the **straight line
  from the ground to the hop's reflection apex** -- exact under the flat-earth
  geometry, and an approximation under curved-earth (good because the D region,
  ~90 km, sits well below the typical F-layer reflection height, ~250-350 km).
* ``earth_model``: ``"flat"`` (planar/equirectangular -- fine to a few 1000 km,
  simple, no pole/date-line handling) or ``"curved"`` (spherical, mean Earth
  radius, great-circle geometry). User-selectable throughout.
* ``height_model`` for the F-layer reflection height per hop: ``"fixed"`` (one
  scalar ``height_km``, independent of frequency) or ``"parabolic"`` (a
  quasi-parabolic layer -- ``foF2_mhz``, ``hmF2_km``, ``ym_km`` -- solved
  iteratively via the secant law for the true oblique reflection height; NOT
  full Croft-Hoogasian ray tracing). Raises if the requested frequency exceeds
  a hop's oblique MUF (no real solution).
* No SNR / link-budget step -- this module stops at absorption (dB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorption import absorption, load_coefficients
from .sza import EARTH_RADIUS_KM, solar_zenith_angle

__all__ = [
    "HopCrossing",
    "great_circle_distance",
    "intermediate_point",
    "incidence_angle",
    "reflection_height",
    "obliquity_factor",
    "hop_crossings",
    "oblique_absorption",
]


@dataclass
class HopCrossing:
    """One D-region crossing of an oblique multi-hop path.

    Attributes
    ----------
    hop : int
        Hop index, 0-based.
    leg : {"up", "down"}
        Ascending (transmitter-side) or descending (receiver-side) crossing.
    lat, lon : float
        Sub-ionospheric point (deg).
    ground_range_km : float
        Distance from the transmitter along the tx-rx path.
    reflection_height_km : float
        F-layer virtual reflection height used for this hop.
    incidence_deg, elevation_deg : float
        Ray incidence angle (from vertical, at the D-region crossing) and
        ground elevation angle.
    """

    hop: int
    leg: str
    lat: float
    lon: float
    ground_range_km: float
    reflection_height_km: float
    incidence_deg: float
    elevation_deg: float


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def great_circle_distance(tx, rx, *, earth_model: str = "curved", R_km: float = EARTH_RADIUS_KM) -> float:
    """Ground range between ``tx`` and ``rx`` (each ``(lat_deg, lon_deg)``), km.

    ``earth_model="curved"`` -- haversine great-circle distance.
    ``earth_model="flat"`` -- equirectangular planar approximation (fine for
    short/medium ranges; do not use near the poles or across the date line).
    """
    lat1, lon1 = np.radians(tx[0]), np.radians(tx[1])
    lat2, lon2 = np.radians(rx[0]), np.radians(rx[1])

    if earth_model == "flat":
        mean_lat = 0.5 * (lat1 + lat2)
        x = (lon2 - lon1) * np.cos(mean_lat)
        y = lat2 - lat1
        return float(R_km * np.hypot(x, y))
    if earth_model == "curved":
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return float(2.0 * R_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
    raise ValueError(f"earth_model must be 'flat' or 'curved', got {earth_model!r}")


def intermediate_point(tx, rx, fraction: float, *, earth_model: str = "curved"):
    """Point a ``fraction`` (0..1) of the way from ``tx`` to ``rx``.

    ``earth_model="curved"`` -- exact great-circle interpolation.
    ``earth_model="flat"`` -- linear interpolation of (lat, lon) (crude planar
    approximation, consistent with :func:`great_circle_distance`'s flat mode).
    """
    lat1, lon1 = np.radians(tx[0]), np.radians(tx[1])
    lat2, lon2 = np.radians(rx[0]), np.radians(rx[1])

    if earth_model == "flat":
        lat = tx[0] + fraction * (rx[0] - tx[0])
        lon = tx[1] + fraction * (rx[1] - tx[1])
        return lat, lon
    if earth_model == "curved":
        d = great_circle_distance(tx, rx, earth_model="curved") / EARTH_RADIUS_KM  # angular distance, rad
        if d < 1e-12:
            return tx[0], tx[1]
        a = np.sin((1 - fraction) * d) / np.sin(d)
        b = np.sin(fraction * d) / np.sin(d)
        x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
        y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
        z = a * np.sin(lat1) + b * np.sin(lat2)
        lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
        lon = np.degrees(np.arctan2(y, x))
        return float(lat), float(lon)
    raise ValueError(f"earth_model must be 'flat' or 'curved', got {earth_model!r}")


def incidence_angle(
    ground_range_km: float,
    height_km: float,
    *,
    earth_model: str = "curved",
    R_km: float = EARTH_RADIUS_KM,
):
    """Ray incidence angle (from vertical) and ground elevation angle for a
    single hop of ``ground_range_km`` reflecting at virtual height ``height_km``.

    Standard secant-law geometry (e.g. Davies, *Ionospheric Radio*). Under the
    straight-line-to-apex approximation this incidence angle is the same
    everywhere along the ray below the apex, so it also applies at the
    D-region crossing height (see module docstring).

    Returns
    -------
    (incidence_deg, elevation_deg)
    """
    if earth_model == "flat":
        elevation = np.degrees(np.arctan2(2.0 * height_km, ground_range_km))
        incidence = 90.0 - elevation
        return float(incidence), float(elevation)
    if earth_model == "curved":
        delta = ground_range_km / (2.0 * R_km)  # half-hop angular range, rad
        elevation = np.degrees(
            np.arctan2(np.cos(delta) - R_km / (R_km + height_km), np.sin(delta))
        )
        incidence = np.degrees(
            np.arcsin(np.clip(R_km / (R_km + height_km) * np.cos(np.radians(elevation)), -1.0, 1.0))
        )
        return float(incidence), float(elevation)
    raise ValueError(f"earth_model must be 'flat' or 'curved', got {earth_model!r}")


# --------------------------------------------------------------------------- #
# Reflection height
# --------------------------------------------------------------------------- #
def reflection_height(
    freq_mhz: float,
    ground_range_km: float,
    *,
    height_model: str = "fixed",
    height_km: float = 300.0,
    foF2_mhz: float | None = None,
    hmF2_km: float | None = None,
    ym_km: float | None = None,
    earth_model: str = "curved",
    R_km: float = EARTH_RADIUS_KM,
    max_iter: int = 50,
    tol_km: float = 1e-3,
):
    """F-layer virtual reflection height (and incidence/elevation) for one hop.

    ``height_model="fixed"`` -- ``height_km`` used as-is, independent of
    frequency; just resolves the geometry at that height.

    ``height_model="parabolic"`` -- a quasi-parabolic F2 layer

        f_N(h)**2 = foF2**2 * (1 - ((h - hmF2) / ym)**2)

    solved self-consistently with the secant law: guess a height, get the
    incidence angle from :func:`incidence_angle`, form the equivalent vertical
    frequency ``f_v = f * cos(incidence)`` (<= ``f``, by the secant law:
    obliquity only ever *raises* the frequency a given layer can reflect, so a
    longer/more-oblique hop never fails where a shorter one succeeds), solve
    the parabolic profile for the height at which ``f_v`` reflects vertically,
    repeat to convergence. Raises ``ValueError`` if ``freq_mhz`` exceeds
    ``foF2_mhz`` (no real reflection height at any obliquity, since ``f_v``
    can never exceed ``freq_mhz`` itself).

    Returns
    -------
    (height_km, incidence_deg, elevation_deg)
    """
    if height_model == "fixed":
        incidence, elevation = incidence_angle(
            ground_range_km, height_km, earth_model=earth_model, R_km=R_km
        )
        return float(height_km), incidence, elevation

    if height_model == "parabolic":
        if foF2_mhz is None or hmF2_km is None or ym_km is None:
            raise ValueError("height_model='parabolic' requires foF2_mhz, hmF2_km, ym_km")
        if freq_mhz > foF2_mhz:
            raise ValueError(
                f"freq_mhz={freq_mhz:g} exceeds foF2={foF2_mhz:g} MHz -- "
                "no reflection even at vertical incidence"
            )
        h = hmF2_km
        incidence = elevation = 0.0
        for _ in range(max_iter):
            incidence, elevation = incidence_angle(
                ground_range_km, h, earth_model=earth_model, R_km=R_km
            )
            f_v = freq_mhz * np.cos(np.radians(incidence))
            ratio = min(f_v / foF2_mhz, 1.0)  # numerical safety at the freq==foF2 boundary
            h_new = hmF2_km - ym_km * np.sqrt(1.0 - ratio**2)
            if abs(h_new - h) < tol_km:
                h = h_new
                break
            h = h_new
        return float(h), incidence, elevation

    raise ValueError(f"height_model must be 'fixed' or 'parabolic', got {height_model!r}")


# --------------------------------------------------------------------------- #
# Secant-law obliquity factor
# --------------------------------------------------------------------------- #
def obliquity_factor(incidence_deg, *, n: float = 1.5):
    """Secant-law obliquity factor ``sec(incidence)**(n + 1)``.

    ``L_oblique(f, incidence) = obliquity_factor(incidence, n) * L_vertical(f)``
    for a vertical-absorption law ``L_vertical(f) ~ f**-n`` (``n = freq_exponent``
    of the absorption model, 1.5 for both bundled models).
    """
    chi = np.radians(np.asarray(incidence_deg, dtype=float))
    return 1.0 / np.cos(chi) ** (n + 1.0)


# --------------------------------------------------------------------------- #
# Hop geometry
# --------------------------------------------------------------------------- #
def hop_crossings(
    tx,
    rx,
    n_hops: int = 1,
    *,
    alt_km: float = 90.0,
    freq_mhz: float | None = None,
    earth_model: str = "curved",
    height_model: str = "fixed",
    height_km: float = 300.0,
    foF2_mhz: float | None = None,
    hmF2_km: float | None = None,
    ym_km: float | None = None,
) -> list[HopCrossing]:
    """The ``2 * n_hops`` D-region crossings of an ``n_hops``-hop tx-rx path.

    Each hop is split at its ground-range midpoint (the reflection apex); the
    two D-region crossings (ascending / descending) are located along the
    straight line from the ground to that apex, at altitude ``alt_km`` (see
    module docstring). ``freq_mhz`` is only used by
    ``height_model="parabolic"``.
    """
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")

    d_total = great_circle_distance(tx, rx, earth_model=earth_model)
    d_hop = d_total / n_hops

    crossings: list[HopCrossing] = []
    for i in range(n_hops):
        h, incidence, elevation = reflection_height(
            freq_mhz if freq_mhz is not None else float("nan"),
            d_hop,
            height_model=height_model,
            height_km=height_km,
            foF2_mhz=foF2_mhz,
            hmF2_km=hmF2_km,
            ym_km=ym_km,
            earth_model=earth_model,
        )
        frac = min(alt_km / h, 1.0) if h > 0 else 1.0  # fraction of the half-hop "rise"
        hop_start = i * d_hop
        for leg, range_from_start in (("up", 0.5 * d_hop * frac),
                                      ("down", d_hop - 0.5 * d_hop * frac)):
            ground_range = hop_start + range_from_start
            lat, lon = intermediate_point(tx, rx, ground_range / d_total, earth_model=earth_model)
            crossings.append(
                HopCrossing(
                    hop=i, leg=leg, lat=lat, lon=lon,
                    ground_range_km=ground_range, reflection_height_km=h,
                    incidence_deg=incidence, elevation_deg=elevation,
                )
            )
    return crossings


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def oblique_absorption(
    tx,
    rx,
    time,
    freq_mhz: float,
    xray_wm2,
    *,
    n_hops: int = 1,
    model: str = "xrap",
    coeffs: dict | None = None,
    grazing: bool | None = None,
    alt_km: float = 90.0,
    H_km: float = 7.0,
    sza_method: str = "noaa",
    earth_model: str = "curved",
    height_model: str = "fixed",
    height_km: float = 300.0,
    foF2_mhz: float | None = None,
    hmF2_km: float | None = None,
    ym_km: float | None = None,
    freq_exponent: float | None = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """Total oblique HF absorption (dB) for an ``n_hops``-hop tx-rx circuit.

    Sums the secant-law-corrected, SZA-dependent vertical absorption
    (:func:`xrap.absorption.absorption`, ``path="oneway"``) over every
    D-region crossing (:func:`hop_crossings`) of the path.

    Parameters
    ----------
    tx, rx : (lat_deg, lon_deg)
        Transmitter and receiver locations.
    time : str, datetime, or array-like of those
        UTC timestamp(s) -- forwarded to :func:`xrap.solar_zenith_angle`.
    freq_mhz : float
        HF operating frequency (MHz).
    xray_wm2 : float or array-like
        GOES 0.1-0.8 nm flux (W m-2), broadcast against ``time``.
    n_hops : int
        Number of ionospheric hops.
    model, coeffs, grazing, alt_km, H_km
        Forwarded to :func:`xrap.absorption.absorption` per crossing.
    sza_method
        Forwarded to :func:`xrap.solar_zenith_angle`.
    earth_model, height_model, height_km, foF2_mhz, hmF2_km, ym_km
        Forwarded to :func:`hop_crossings` / :func:`reflection_height`.
    freq_exponent : float, optional
        Exponent for :func:`obliquity_factor`; defaults to the model's
        ``freq_exponent`` (1.5).
    clip_negative : bool
        Clip the summed result to >= 0 dB.

    Returns
    -------
    numpy.ndarray
        Total absorption (dB) along the path, broadcast over ``time`` /
        ``xray_wm2``.
    """
    c = coeffs if coeffs is not None else load_coefficients(model)
    n = c["freq_exponent"] if freq_exponent is None else freq_exponent

    crossings = hop_crossings(
        tx, rx, n_hops, alt_km=alt_km, freq_mhz=freq_mhz, earth_model=earth_model,
        height_model=height_model, height_km=height_km,
        foF2_mhz=foF2_mhz, hmF2_km=hmF2_km, ym_km=ym_km,
    )

    total = 0.0
    for cr in crossings:
        chi = solar_zenith_angle(cr.lat, cr.lon, time, method=sza_method)
        L_v = absorption(
            freq_mhz, chi, xray_wm2, model=model, coeffs=c, path="oneway",
            grazing=grazing, alt_km=alt_km, H_km=H_km, clip_negative=False,
        )
        total = total + obliquity_factor(cr.incidence_deg, n=n) * L_v

    total = np.asarray(total, dtype=float)
    if clip_negative:
        total = np.maximum(total, 0.0)
    return total
