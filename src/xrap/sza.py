"""Solar geometry helpers.

The absorption model is driven by the solar zenith angle (SZA, ``chi``) at the
point where the HF ray traverses the D region, plus -- near the terminator --
the *grazing* geometry, which depends on the altitude of that point.

Two things vary with altitude ``h``:

* **Illumination cut-off.** The Sun's direction (hence the true geometric SZA)
  is effectively altitude-independent -- solar parallax over 100 km is
  ~1e-4 deg. What *does* change is that a point at height ``h`` stays in
  sunlight until the true SZA reaches ``90 deg + horizon_dip(h)`` (the solid
  Earth stops shadowing it). For ``h = 90 km`` that extends the sunlit region
  to SZA ~ 99.6 deg. Use :func:`is_sunlit` / :func:`terminator_sza`.

* **Slant column.** For SZA approaching and beyond 90 deg, ``sec(chi)``
  diverges and is unphysical. Replace it with the Chapman grazing-incidence
  function :func:`chapman_function` ``Ch(X, chi)`` with ``X = (R + h) / H``.

So :func:`solar_zenith_angle` returns the true geometric SZA and takes
``alt_km`` (forwarded to the astropy path for parallax/refraction, and carried
through for the caller's convenience); the height-critical grazing behaviour
lives in the helpers below and is what the absorption model should consume near
SZA ~ 90 deg.

TODO / decisions to confirm:
* Reference altitude of the absorbing layer (default 90 km here).
* Neutral scale height ``H`` for the Chapman function (7 km placeholder).
* Whether to use the ground point or the sub-ionospheric (ray-piercing) point.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.special import erfcx

__all__ = [
    "solar_zenith_angle",
    "solar_declination",
    "equation_of_time",
    "horizon_dip",
    "terminator_sza",
    "is_sunlit",
    "chapman_function",
    "EARTH_RADIUS_KM",
]

#: Mean Earth radius used for the grazing geometry (km).
EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- #
# Solar zenith angle
# --------------------------------------------------------------------------- #
def solar_zenith_angle(
    lat: float | Iterable[float],
    lon: float | Iterable[float],
    time,
    *,
    alt_km: float | Iterable[float] = 0.0,
    method: str = "noaa",
    refraction: bool = False,
) -> np.ndarray:
    """Solar zenith angle in degrees.

    Parameters
    ----------
    lat, lon : float or array-like
        Geographic latitude and East-positive longitude in degrees. Broadcast
        against ``time`` (and each other) with :func:`numpy.broadcast_arrays`,
        so either pass scalars, or arrays with matching/broadcastable shapes.
    time : str, datetime, or array-like of those
        UTC timestamp(s). Parsed with :func:`pandas.to_datetime` (``utc=True``).
    alt_km : float or array-like, default 0.0
        Altitude of the point. Affects the *returned angle* only through the
        astropy path (diurnal parallax + optional refraction), where the effect
        is < 0.01 deg. Its real importance is the grazing regime -- see
        :func:`is_sunlit` and :func:`chapman_function`, which take ``alt_km``
        directly.
    method : {"noaa", "astropy"}
        ``"noaa"`` -- analytic NOAA low-precision solar position (fast, fully
        vectorised, no external ephemeris). ``"astropy"`` -- full ephemeris via
        :mod:`astropy.coordinates` (accurate to arcsec; honours ``alt_km`` and
        ``refraction`` rigorously).
    refraction : bool, default False
        Apply atmospheric-refraction correction to the apparent elevation.

    Returns
    -------
    numpy.ndarray
        SZA in degrees, broadcast shape of the inputs. Values > 90 mean the Sun
        is below the *astronomical* horizon; compare against
        :func:`terminator_sza` to know whether a layer at ``alt_km`` is still
        lit.
    """
    times = pd.to_datetime(np.atleast_1d(time), utc=True)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt_km = np.asarray(alt_km, dtype=float)

    if method == "noaa":
        return _sza_noaa(lat, lon, times, refraction)
    if method == "astropy":
        return _sza_astropy(lat, lon, times, alt_km, refraction)
    raise ValueError(f"unknown method {method!r}")


def _solar_geometry(jd: np.ndarray):
    """NOAA low-precision solar geometry.

    Parameters
    ----------
    jd : ndarray
        Julian date(s), UTC.

    Returns
    -------
    decl : ndarray
        Apparent solar declination (radians).
    eqtime : ndarray
        Equation of time (minutes).
    """
    t = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0  # mean longitude
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)             # mean anomaly
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)        # orbit eccentricity
    mr = np.radians(m)

    # Sun's equation of centre and true/apparent ecliptic longitude
    centre = (
        np.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mr) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mr) * 0.000289
    )
    true_long = l0 + centre
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    # Obliquity of the ecliptic (corrected)
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = np.radians(eps0 + 0.00256 * np.cos(np.radians(omega)))

    decl = np.arcsin(np.sin(eps) * np.sin(np.radians(app_long)))

    y = np.tan(eps / 2.0) ** 2
    l0r = np.radians(l0)
    eqtime = 4.0 * np.degrees(
        y * np.sin(2 * l0r)
        - 2.0 * e * np.sin(mr)
        + 4.0 * e * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r)
        - 1.25 * e * e * np.sin(2 * mr)
    )
    return decl, eqtime


def _sza_noaa(lat, lon, times, refraction) -> np.ndarray:
    """SZA via the NOAA analytic algorithm (Meeus low-precision form)."""
    jd = np.asarray(times.to_julian_date(), dtype=float)
    minutes = np.asarray(
        times.hour * 60.0 + times.minute + times.second / 60.0 + times.microsecond / 6.0e7,
        dtype=float,
    )
    decl, eqtime = _solar_geometry(jd)

    lat_b, lon_b, decl_b, eqtime_b, min_b = np.broadcast_arrays(
        lat, lon, decl, eqtime, minutes
    )

    true_solar_min = (min_b + eqtime_b + 4.0 * lon_b) % 1440.0  # UTC -> no tz offset
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)

    latr = np.radians(lat_b)
    cos_zen = np.clip(
        np.sin(latr) * np.sin(decl_b) + np.cos(latr) * np.cos(decl_b) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    zenith = np.degrees(np.arccos(cos_zen))

    if refraction:
        elev = 90.0 - zenith
        zenith = 90.0 - (elev + _refraction_correction(elev))
    return np.asarray(zenith)


def _sza_astropy(lat, lon, times, alt_km, refraction) -> np.ndarray:
    """SZA via astropy's AltAz transform of the Sun."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time

    try:  # get_body is preferred since astropy 5.3
        from astropy.coordinates import get_body

        def _sun(t, loc):
            return get_body("sun", t, loc)
    except ImportError:  # pragma: no cover - old astropy
        from astropy.coordinates import get_sun

        def _sun(t, loc):
            return get_sun(t)

    loc = EarthLocation(
        lat=np.asarray(lat) * u.deg,
        lon=np.asarray(lon) * u.deg,
        height=np.asarray(alt_km) * u.km,
    )
    t = Time(times.tz_convert("UTC").tz_localize(None).to_pydatetime(), scale="utc")

    frame = AltAz(
        obstime=t,
        location=loc,
        pressure=(101325.0 * u.Pa if refraction else 0.0 * u.Pa),
    )
    alt = _sun(t, loc).transform_to(frame).alt.to_value(u.deg)
    return np.asarray(90.0 - alt)


def _refraction_correction(elev_deg: np.ndarray) -> np.ndarray:
    """Atmospheric-refraction correction to add to the true elevation (deg).

    NOAA piecewise fit; input and output in degrees.
    """
    elev = np.asarray(elev_deg, dtype=float)
    te = np.tan(np.radians(np.where(np.abs(elev) < 1e-6, 1e-6, elev)))
    high = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    low = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    very_low = -20.774 / te
    arcsec = np.where(
        elev > 85.0,
        0.0,
        np.where(elev > 5.0, high, np.where(elev > -0.575, low, very_low)),
    )
    return arcsec / 3600.0


def solar_declination(time) -> np.ndarray:
    """Apparent solar declination in degrees."""
    times = pd.to_datetime(np.atleast_1d(time), utc=True)
    decl, _ = _solar_geometry(np.asarray(times.to_julian_date(), dtype=float))
    return np.degrees(decl)


def equation_of_time(time) -> np.ndarray:
    """Equation of time in minutes."""
    times = pd.to_datetime(np.atleast_1d(time), utc=True)
    _, eqtime = _solar_geometry(np.asarray(times.to_julian_date(), dtype=float))
    return np.asarray(eqtime)


# --------------------------------------------------------------------------- #
# Height-dependent grazing geometry
# --------------------------------------------------------------------------- #
def horizon_dip(alt_km, R_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Angular dip of the geometric horizon for a point at ``alt_km`` (deg).

    Sunlight reaches that altitude for true SZA up to ``90 + horizon_dip``.
    """
    h = np.asarray(alt_km, dtype=float)
    return np.degrees(np.arccos(R_km / (R_km + h)))


def terminator_sza(alt_km, R_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Largest true SZA (deg) at which a layer at ``alt_km`` is still sunlit."""
    return 90.0 + horizon_dip(alt_km, R_km)


def is_sunlit(sza_deg, alt_km, R_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Boolean mask: is a layer at ``alt_km`` in direct sunlight at this SZA?"""
    return np.asarray(sza_deg, dtype=float) < terminator_sza(alt_km, R_km)


def chapman_function(
    sza_deg,
    alt_km,
    H_km: float = 7.0,
    R_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Chapman grazing-incidence function ``Ch(X, chi)``.

    The ratio (slant column) / (vertical column) for a plane-stratified
    atmosphere on a curved Earth. Replaces ``sec(chi)`` and stays finite as
    ``chi`` approaches and crosses 90 deg. Smith & Smith (1972) approximation,
    evaluated with :func:`scipy.special.erfcx` for numerical stability.

    Parameters
    ----------
    sza_deg : float or array-like
        True solar zenith angle (deg).
    alt_km : float or array-like
        Altitude of the absorbing layer. Enters as ``X = (R_km + alt_km) / H_km``.
    H_km : float, default 7.0
        Neutral scale height near the layer. **Placeholder** -- confirm the
        value appropriate for the D region.
    R_km : float
        Earth radius.

    Returns
    -------
    numpy.ndarray
        ``Ch(X, chi)`` -> ``sec(chi)`` for small ``chi``; ~ ``sqrt(pi X / 2)``
        at ``chi = 90 deg``; grows rapidly (and becomes unphysical) well past
        the terminator, so gate with :func:`is_sunlit` first.
    """
    chi = np.radians(np.asarray(sza_deg, dtype=float))
    x = (R_km + np.asarray(alt_km, dtype=float)) / H_km
    cos_chi = np.cos(chi)
    sin_chi = np.sin(chi)
    y = np.sqrt(x / 2.0) * np.abs(cos_chi)

    ch_day = np.sqrt(np.pi * x / 2.0) * erfcx(y)
    ch_night = np.sqrt(2.0 * np.pi * x) * (
        np.sqrt(sin_chi) * np.exp(np.clip(x * (1.0 - sin_chi), None, 300.0))
        - 0.5 * erfcx(y)
    )
    return np.where(cos_chi >= 0.0, ch_day, ch_night)
