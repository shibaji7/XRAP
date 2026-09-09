"""Tests for xrap.sza."""

from __future__ import annotations

import numpy as np
import pytest

from xrap.sza import (
    chapman_function,
    equation_of_time,
    horizon_dip,
    is_sunlit,
    solar_declination,
    solar_zenith_angle,
    terminator_sza,
)


# --------------------------------------------------------------------------- #
# Solar zenith angle
# --------------------------------------------------------------------------- #
def test_local_noon_near_subsolar_point():
    """At ~local solar noon on an equinox, SZA ~ |latitude|."""
    chi = solar_zenith_angle(lat=0.0, lon=0.0, time="2020-03-20T12:07Z", method="noaa")
    assert chi.shape == (1,)
    assert chi[0] == pytest.approx(0.0, abs=2.0)


def test_night_side_exceeds_90():
    chi = solar_zenith_angle(lat=40.0, lon=-105.0, time="2017-09-06T06:00Z", method="noaa")
    assert chi[0] > 90.0


def test_vectorized_over_time():
    times = np.array(["2017-09-06T10:00", "2017-09-06T12:00", "2017-09-06T14:00"],
                     dtype="datetime64[ns]")
    chi = solar_zenith_angle(40.0, -105.0, times, method="noaa")
    assert chi.shape == (3,)
    # SZA is smallest nearest local noon (~19 UT at -105 deg lon), so it should
    # be decreasing across 10-14 UT.
    assert chi[0] > chi[1] > chi[2]


def test_broadcast_latitude_array():
    chi = solar_zenith_angle(np.array([0.0, 30.0, 60.0]), 0.0,
                             "2020-03-20T12:07Z", method="noaa")
    assert chi.shape == (3,)
    assert np.all(np.diff(chi) > 0)  # SZA grows with |lat| at subsolar noon


def test_astropy_and_noaa_agree():
    kw = dict(lat=52.0, lon=13.0, time="2015-03-20T09:45Z")
    assert solar_zenith_angle(method="astropy", **kw) == pytest.approx(
        solar_zenith_angle(method="noaa", **kw), abs=0.5
    )


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        solar_zenith_angle(0.0, 0.0, "2020-01-01T00:00Z", method="bogus")


def test_declination_within_obliquity():
    d = solar_declination(["2020-03-20", "2020-06-21", "2020-12-21"])
    assert d[0] == pytest.approx(0.0, abs=1.0)
    assert d[1] == pytest.approx(23.44, abs=0.3)
    assert d[2] == pytest.approx(-23.44, abs=0.3)


def test_equation_of_time_magnitude():
    eot = equation_of_time(["2020-02-11", "2020-11-03"])
    assert abs(eot[0]) < 20.0 and abs(eot[1]) < 20.0


# --------------------------------------------------------------------------- #
# Height-dependent grazing geometry
# --------------------------------------------------------------------------- #
def test_horizon_dip_zero_at_ground():
    assert horizon_dip(0.0) == pytest.approx(0.0)


def test_terminator_extends_with_altitude():
    # A 90 km layer stays sunlit to SZA ~ 99.6 deg.
    assert terminator_sza(90.0) == pytest.approx(99.6, abs=0.3)
    assert terminator_sza(90.0) > terminator_sza(0.0)


def test_is_sunlit_across_terminator():
    assert is_sunlit(95.0, alt_km=90.0)          # below 99.6 deg -> lit
    assert not is_sunlit(95.0, alt_km=0.0)       # ground point is dark


def test_chapman_reduces_to_secant_for_small_sza():
    # Ch is slightly *below* sec(chi) because the atmosphere is curved, not
    # plane-parallel; the gap grows with chi but stays sub-percent well away
    # from the terminator.
    chi = np.array([0.0, 30.0, 60.0])
    ch = chapman_function(chi, alt_km=90.0)
    sec = 1.0 / np.cos(np.radians(chi))
    np.testing.assert_allclose(ch, sec, rtol=2e-2)
    assert np.all(ch <= sec + 1e-9)


def test_chapman_finite_and_monotonic_through_terminator():
    chi = np.array([80.0, 88.0, 90.0, 95.0, 98.0])
    ch = chapman_function(chi, alt_km=90.0)
    assert np.all(np.isfinite(ch))
    assert np.all(np.diff(ch) > 0)
    # at exactly 90 deg, Ch ~ sqrt(pi X / 2)
    x = (6371.0 + 90.0) / 7.0
    assert chapman_function(90.0, alt_km=90.0) == pytest.approx(np.sqrt(np.pi * x / 2), rel=0.05)
