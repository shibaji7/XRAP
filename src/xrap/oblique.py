"""Oblique-path (transmitter-to-receiver) HF absorption via the secant law.

:func:`xrap.absorption.absorption` gives **vertical** D-region absorption. A
real HF circuit is oblique: the wave leaves the transmitter at some elevation
angle, is refracted back by the F layer one or more hops later, and crosses
the D region twice per hop (once going up, once coming down) -- each crossing
generally at a different location (hence a different solar zenith angle) and
at non-vertical incidence.

D-region absorption is *non-deviative*: the plasma there is far too thin to bend
the ray, so the wave crosses it in a straight line at the actual wave frequency
``f``. From the absorption coefficient ``kappa ~ N nu / (mu (omega**2 + nu**2))``
and ``ds = dh / cos(theta)``, an oblique crossing at incidence ``chi_D`` (angle
from vertical **at the absorbing height**) accumulates exactly ``sec(chi_D)``
times the vertical absorption at the same frequency::

    L_oblique(f, chi_D) = sec(chi_D) * L_vertical(f)                        (1)

independent of the exponent ``n`` in ``L_vertical ~ f**-n``. This is also the
structure of ITU-R P.533-14 eq. (20) (one ``sec i`` factor, actual-frequency
``(f + f_L)**-2``; the equivalent-vertical-frequency ``f_v = f cos i`` enters
only through a bounded penetration factor, not as a power of ``cos``).

Two ways to get this wrong, both of which an earlier version of this module
did: (a) evaluate the incidence angle at the F-layer reflection height rather
than at the D region -- on a curved Earth the ray's angle from local vertical
grows toward the ground (Bouguer: ``r sin(theta) = const``), e.g. 72 deg at
300 km but 80 deg at 90 km on a 3000 km hop; (b) multiply ``sec`` by a
Martyn-style equivalent-frequency shift ``f -> f cos(chi)`` as well, which
double-counts the obliquity and gives ``sec(chi)**(n+1)``.

This module supplies the geometry (:func:`hop_crossings`) to locate every
D-region crossing of an ``n_hops``-hop path and its incidence angle *at the
D-region height*, then sums the per-crossing, obliquity-corrected,
SZA-dependent vertical absorption (:func:`oblique_absorption`).

Scope and approximations
-------------------------
* Non-deviative D-region absorption only (same regime as the vertical models);
  no F-region deviative-absorption or MUF-cusp enhancement.
* Each hop's ray is the **straight chord from the ground to its reflection
  apex** (the usual virtual-mirror model). Both the D-region incidence angle
  and the two crossing locations follow *exactly* from that chord on either
  Earth model.
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
* The vertical models are not validated far from their 30 MHz riometer
  calibration, and neither saturates; pass ``max_db`` to scale the summed time
  series' peak down to it (see :func:`oblique_absorption`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorption import _apply_max_db, absorption
from .sza import EARTH_RADIUS_KM, solar_zenith_angle

__all__ = [
    "HopCrossing",
    "great_circle_distance",
    "intermediate_point",
    "incidence_angle",
    "crossing_geometry",
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
    incidence_deg : float
        Ray incidence angle from vertical **at the D-region crossing height**
        (``alt_km``) -- the angle the secant law needs.
    elevation_deg : float
        Ground elevation angle of the hop.
    mirror_incidence_deg : float
        Incidence angle at the F-layer reflection point (smaller than
        ``incidence_deg`` on a curved Earth; kept for reference only).
    """

    hop: int
    leg: str
    lat: float
    lon: float
    ground_range_km: float
    reflection_height_km: float
    incidence_deg: float
    elevation_deg: float
    mirror_incidence_deg: float = float("nan")


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
    """Incidence angle **at the reflection point** and ground elevation angle
    for a single hop of ``ground_range_km`` reflecting at virtual height
    ``height_km``.

    Standard secant-law geometry (e.g. Davies, *Ionospheric Radio*). The
    returned incidence angle is the one at the *mirror* (height ``height_km``),
    which is what the reflection condition needs. It is **not** the angle at
    the D region: on a curved Earth the ray's angle from local vertical varies
    with height (Bouguer's invariant), so use :func:`crossing_geometry` for the
    absorbing layer.

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


def crossing_geometry(
    elevation_deg: float,
    alt_km: float,
    *,
    earth_model: str = "curved",
    R_km: float = EARTH_RADIUS_KM,
):
    """Where, and at what incidence, a ray of ground elevation ``elevation_deg``
    crosses height ``alt_km`` on its way up (straight-chord ray).

    * flat: incidence = ``90 - elevation``; ground range = ``alt_km / tan(elevation)``.
    * curved: Bouguer's invariant gives the incidence angle at radius
      ``R + alt_km``,  ``sin(chi) = R cos(elevation) / (R + alt_km)``, and the
      triangle (Earth centre)-(transmitter)-(crossing point) gives the central
      angle ``psi = 90 deg - elevation - chi``, i.e. ground range ``R * psi``.

    Returns
    -------
    (incidence_deg, ground_range_km)
        Incidence angle from vertical at ``alt_km``, and the ground distance
        from the transmitter to the crossing point.
    """
    beta = np.radians(elevation_deg)
    if earth_model == "flat":
        return float(90.0 - elevation_deg), float(alt_km / np.tan(beta))
    if earth_model == "curved":
        chi = np.arcsin(np.clip(R_km * np.cos(beta) / (R_km + alt_km), -1.0, 1.0))
        psi = max(np.pi / 2.0 - beta - chi, 0.0)
        return float(np.degrees(chi)), float(R_km * psi)
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
def obliquity_factor(incidence_deg, *, exponent: float = 1.0):
    """Secant obliquity factor ``sec(incidence)**exponent``.

    ``L_oblique(f, chi_D) = obliquity_factor(chi_D) * L_vertical(f)`` with
    ``chi_D`` the incidence angle **at the D-region height** and ``exponent = 1``
    (non-deviative path-length secant at the actual frequency; see the module
    docstring). Other exponents are available for experiments only.
    """
    chi = np.radians(np.asarray(incidence_deg, dtype=float))
    return 1.0 / np.cos(chi) ** exponent


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
    two D-region crossings (ascending / descending) are where the straight
    chord from the ground to that apex passes through ``alt_km``
    (:func:`crossing_geometry`), each carrying the incidence angle at that
    height. ``freq_mhz`` is only used by ``height_model="parabolic"``.
    """
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")

    d_total = great_circle_distance(tx, rx, earth_model=earth_model)
    d_hop = d_total / n_hops

    crossings: list[HopCrossing] = []
    for i in range(n_hops):
        h, mirror_incidence, elevation = reflection_height(
            freq_mhz if freq_mhz is not None else float("nan"),
            d_hop,
            height_model=height_model,
            height_km=height_km,
            foF2_mhz=foF2_mhz,
            hmF2_km=hmF2_km,
            ym_km=ym_km,
            earth_model=earth_model,
        )
        incidence, up_range = crossing_geometry(elevation, alt_km, earth_model=earth_model)
        up_range = min(up_range, 0.5 * d_hop)  # apex below alt_km: cross at the apex
        hop_start = i * d_hop
        for leg, range_from_start in (("up", up_range), ("down", d_hop - up_range)):
            ground_range = hop_start + range_from_start
            lat, lon = intermediate_point(tx, rx, ground_range / d_total, earth_model=earth_model)
            crossings.append(
                HopCrossing(
                    hop=i, leg=leg, lat=lat, lon=lon,
                    ground_range_km=ground_range, reflection_height_km=h,
                    incidence_deg=incidence, elevation_deg=elevation,
                    mirror_incidence_deg=mirror_incidence,
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
    obliquity_exponent: float = 1.0,
    max_db: float | None = None,
    max_db_mode: str = "scale",
    freq_saturation_mhz: float | str | None = "default",
    clip_negative: bool = True,
) -> np.ndarray:
    """Total oblique HF absorption (dB) for an ``n_hops``-hop tx-rx circuit.

    Sums the obliquity-corrected, SZA-dependent vertical absorption
    (:func:`xrap.absorption.absorption`, ``path="oneway"``) over every
    D-region crossing (:func:`hop_crossings`) of the path:
    ``sum_i sec(chi_i)**obliquity_exponent * L_vertical(f, SZA_i, flux)``, with
    ``chi_i`` the incidence angle at the D-region height.

    Parameters
    ----------
    tx, rx : (lat_deg, lon_deg)
        Transmitter and receiver locations.
    time : str, datetime, or array-like of those
        UTC timestamp(s) -- forwarded to :func:`xrap.solar_zenith_angle`.
    freq_mhz : float
        HF operating frequency (MHz).
    xray_wm2 : float or array-like
        GOES 0.1-0.8 nm flux (W m-2) **on the NOAA flare-class scale**
        (``X9.3 = 9.3e-4``) -- what :func:`xrap.fetch_goes_xrs` returns by
        default. Broadcast against ``time``.
    n_hops : int
        Number of ionospheric hops.
    model, coeffs, grazing, alt_km, H_km
        Forwarded to :func:`xrap.absorption.absorption` per crossing.
    sza_method
        Forwarded to :func:`xrap.solar_zenith_angle`.
    earth_model, height_model, height_km, foF2_mhz, hmF2_km, ym_km
        Forwarded to :func:`hop_crossings` / :func:`reflection_height`.
    obliquity_exponent : float
        Exponent of the secant factor; 1 (default) is the non-deviative
        path-length law (see the module docstring).
    max_db : float, optional
        Limit on the **summed** total (dB). The vertical models are linear in
        flux and unbounded, so a strong flare on a long, low-frequency path can
        reach thousands of dB -- far beyond any circuit's margin. With
        ``max_db_mode="scale"`` (default) the **whole returned time series is
        multiplied by one factor so its peak equals** ``max_db`` (only if the peak
        exceeds it; :func:`xrap.absorption.peak_scale`): the shape is preserved,
        there is no flat top. The factor depends on the window you pass, quiet
        values are rescaled too, and series scaled separately are no longer
        comparable. ``None`` (default) leaves the raw value.
    max_db_mode : {"scale", "clip"}
        ``"scale"`` (default) or ``"clip"`` (hard cut at ``max_db``; then
        ``result >= max_db`` means "blackout").
    freq_saturation_mhz : float, None or "default"
        Low-frequency saturation of the vertical law, forwarded to
        :func:`xrap.absorption.absorption` (XRAP default 5 MHz; ``None`` disables).
    clip_negative : bool
        Clip the summed result to >= 0 dB.

    Returns
    -------
    numpy.ndarray
        Total absorption (dB) along the path, broadcast over ``time`` /
        ``xray_wm2``.
    """
    crossings = hop_crossings(
        tx, rx, n_hops, alt_km=alt_km, freq_mhz=freq_mhz, earth_model=earth_model,
        height_model=height_model, height_km=height_km,
        foF2_mhz=foF2_mhz, hmF2_km=hmF2_km, ym_km=ym_km,
    )

    total = 0.0
    for cr in crossings:
        chi = solar_zenith_angle(cr.lat, cr.lon, time, method=sza_method)
        L_v = absorption(
            freq_mhz, chi, xray_wm2, model=model, coeffs=coeffs, path="oneway",
            grazing=grazing, alt_km=alt_km, H_km=H_km, clip_negative=False,
            freq_saturation_mhz=freq_saturation_mhz,
        )
        total = total + obliquity_factor(cr.incidence_deg, exponent=obliquity_exponent) * L_v

    total = np.asarray(total, dtype=float)
    if clip_negative:
        total = np.maximum(total, 0.0)
    return _apply_max_db(total, max_db, max_db_mode)
