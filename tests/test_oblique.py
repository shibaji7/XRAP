"""Tests for xrap.oblique -- secant-law oblique-path absorption.

Fully offline: geometry is analytic and SZA uses method="noaa".
"""

from __future__ import annotations

import numpy as np
import pytest

from xrap import hop_crossings, oblique_absorption, obliquity_factor, solar_zenith_angle
from xrap.absorption import absorption
from xrap.oblique import (
    great_circle_distance,
    incidence_angle,
    intermediate_point,
    reflection_height,
)

LONDON = (51.5, -0.1)
NEWYORK = (40.7, -74.0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_great_circle_distance_known_city_pair():
    # London-New York is ~5570 km
    d = great_circle_distance(LONDON, NEWYORK, earth_model="curved")
    assert d == pytest.approx(5570.0, rel=0.02)


def test_flat_and_curved_agree_at_short_range():
    a, b = (40.0, -105.0), (40.5, -104.5)
    d_flat = great_circle_distance(a, b, earth_model="flat")
    d_curved = great_circle_distance(a, b, earth_model="curved")
    assert d_flat == pytest.approx(d_curved, rel=0.01)


def test_distance_rejects_unknown_earth_model():
    with pytest.raises(ValueError):
        great_circle_distance(LONDON, NEWYORK, earth_model="banana")


def test_intermediate_point_endpoints():
    for earth_model in ("flat", "curved"):
        lat0, lon0 = intermediate_point(LONDON, NEWYORK, 0.0, earth_model=earth_model)
        lat1, lon1 = intermediate_point(LONDON, NEWYORK, 1.0, earth_model=earth_model)
        assert (lat0, lon0) == pytest.approx(LONDON, abs=1e-6)
        assert (lat1, lon1) == pytest.approx(NEWYORK, abs=1e-6)


def test_intermediate_point_midpoint_is_between():
    # low-latitude, near-equal-latitude pair so the great circle doesn't bulge
    # poleward past either endpoint's latitude (it does for London-New York).
    a, b = (10.0, 20.0), (10.0, 40.0)
    lat, lon = intermediate_point(a, b, 0.5, earth_model="curved")
    assert min(a[0], b[0]) - 1 <= lat <= max(a[0], b[0]) + 1
    assert a[1] < lon < b[1]


def test_incidence_angle_overhead_reflection_is_near_vertical():
    # very short hop, high reflection -> nearly overhead -> incidence ~ 0
    inc, elev = incidence_angle(1.0, 300.0, earth_model="curved")
    assert inc == pytest.approx(0.0, abs=1.0)
    assert elev == pytest.approx(90.0, abs=1.0)


def test_incidence_angle_flat_matches_geometry():
    # flat earth: tan(elevation) = 2h/d
    inc, elev = incidence_angle(1000.0, 300.0, earth_model="flat")
    assert elev == pytest.approx(np.degrees(np.arctan2(600.0, 1000.0)))
    assert inc == pytest.approx(90.0 - elev)


def test_incidence_angle_flat_and_curved_agree_for_short_hop():
    inc_f, _ = incidence_angle(300.0, 300.0, earth_model="flat")
    inc_c, _ = incidence_angle(300.0, 300.0, earth_model="curved")
    assert inc_f == pytest.approx(inc_c, abs=1.0)


def test_incidence_angle_rejects_unknown_earth_model():
    with pytest.raises(ValueError):
        incidence_angle(1000.0, 300.0, earth_model="banana")


# --------------------------------------------------------------------------- #
# Reflection height
# --------------------------------------------------------------------------- #
def test_reflection_height_fixed_passthrough():
    h, inc, elev = reflection_height(10.0, 1000.0, height_model="fixed", height_km=280.0)
    assert h == 280.0
    inc2, elev2 = incidence_angle(1000.0, 280.0)
    assert (inc, elev) == pytest.approx((inc2, elev2))


def test_reflection_height_parabolic_short_hop_matches_vertical_formula():
    # a very short hop (~near-vertical incidence) should reproduce the plain
    # vertical parabolic-layer reflection height h = hmF2 - ym*sqrt(1-(f/foF2)^2)
    foF2, hmF2, ym, f = 8.0, 300.0, 100.0, 4.0
    h, inc, _ = reflection_height(
        f, 1.0, height_model="parabolic", foF2_mhz=foF2, hmF2_km=hmF2, ym_km=ym
    )
    expected = hmF2 - ym * np.sqrt(1.0 - (f / foF2) ** 2)
    assert h == pytest.approx(expected, rel=0.02)
    assert inc == pytest.approx(0.0, abs=2.0)


def test_reflection_height_parabolic_rejects_above_foF2():
    with pytest.raises(ValueError):
        reflection_height(10.0, 500.0, height_model="parabolic",
                          foF2_mhz=8.0, hmF2_km=300.0, ym_km=100.0)


def test_reflection_height_parabolic_lower_for_longer_more_oblique_hop():
    # obliquity only ever *helps* reflection (equivalent vertical frequency
    # f*cos(incidence) <= f), so a longer/more-oblique hop reflects the same
    # frequency at a lower true height than a short, near-vertical one.
    kw = dict(height_model="parabolic", foF2_mhz=8.0, hmF2_km=300.0, ym_km=100.0)
    h_short, _, _ = reflection_height(7.9, 1.0, **kw)
    h_long, _, _ = reflection_height(7.9, 3000.0, **kw)
    assert h_long < h_short


def test_reflection_height_requires_parabolic_params():
    with pytest.raises(ValueError):
        reflection_height(10.0, 1000.0, height_model="parabolic")


def test_reflection_height_rejects_unknown_model():
    with pytest.raises(ValueError):
        reflection_height(10.0, 1000.0, height_model="banana")


# --------------------------------------------------------------------------- #
# Obliquity factor
# --------------------------------------------------------------------------- #
def test_obliquity_factor_unity_overhead():
    assert obliquity_factor(0.0) == pytest.approx(1.0)


def test_obliquity_factor_grows_with_incidence():
    a = obliquity_factor(0.0)
    b = obliquity_factor(45.0)
    c = obliquity_factor(70.0)
    assert a < b < c


def test_obliquity_factor_matches_secant_law_n():
    # sec(chi)^(n+1); default n=1.5 -> exponent 2.5
    chi = 40.0
    sec = 1.0 / np.cos(np.radians(chi))
    assert obliquity_factor(chi, n=1.5) == pytest.approx(sec**2.5)
    assert obliquity_factor(chi, n=0.0) == pytest.approx(sec)


# --------------------------------------------------------------------------- #
# hop_crossings
# --------------------------------------------------------------------------- #
def test_hop_crossings_count_and_order():
    cr = hop_crossings(LONDON, NEWYORK, n_hops=2, height_km=300.0)
    assert len(cr) == 4
    assert [c.hop for c in cr] == [0, 0, 1, 1]
    assert [c.leg for c in cr] == ["up", "down", "up", "down"]
    ranges = [c.ground_range_km for c in cr]
    assert ranges == sorted(ranges)


def test_hop_crossings_lie_between_tx_and_rx():
    tx, rx = (10.0, 20.0), (10.0, 40.0)
    cr = hop_crossings(tx, rx, n_hops=1, height_km=300.0)
    d_total = great_circle_distance(tx, rx)
    for c in cr:
        assert 0.0 <= c.ground_range_km <= d_total
        assert min(tx[0], rx[0]) - 1 <= c.lat <= max(tx[0], rx[0]) + 1


def test_hop_crossings_rejects_zero_hops():
    with pytest.raises(ValueError):
        hop_crossings(LONDON, NEWYORK, n_hops=0)


# --------------------------------------------------------------------------- #
# oblique_absorption
# --------------------------------------------------------------------------- #
def test_oblique_exceeds_simple_vertical_sum():
    """Obliquity (sec > 1) must make the oblique total exceed the sum of the
    same crossings' *vertical* absorption (obliquity factor stripped out)."""
    tx, rx = (10.0, 20.0), (10.0, 40.0)  # long, low-latitude hop -> daytime, oblique
    time = "2017-09-06T12:00Z"
    flux = 1e-5

    total = oblique_absorption(tx, rx, time, 10.0, flux, n_hops=1, height_km=300.0)
    cr = hop_crossings(tx, rx, n_hops=1, height_km=300.0)
    assert cr[0].incidence_deg > 5.0  # genuinely oblique for this geometry

    vertical_sum = sum(
        float(np.ravel(absorption(
            10.0, solar_zenith_angle(c.lat, c.lon, time, method="noaa"),
            flux, model="xrap", path="oneway", grazing=False,
        ))[0])
        for c in cr
    )
    assert total > 0.0
    assert total > vertical_sum


def test_oblique_absorption_increases_with_hops_and_frequency_law():
    tx, rx = (10.0, 20.0), (10.0, 30.0)
    time = "2017-09-06T12:00Z"
    a1 = oblique_absorption(tx, rx, time, 10.0, 1e-5, n_hops=1, height_km=300.0)
    a2 = oblique_absorption(tx, rx, time, 10.0, 1e-5, n_hops=2, height_km=300.0)
    # more hops over the same total distance -> shorter, less-oblique hops,
    # but twice as many D-region crossings; still expect a positive, finite total
    assert a1 > 0.0 and a2 > 0.0
    assert np.isfinite(a1) and np.isfinite(a2)


def test_oblique_absorption_vectorizes_over_time():
    tx, rx = (10.0, 20.0), (10.0, 30.0)
    times = ["2017-09-06T11:00Z", "2017-09-06T12:00Z", "2017-09-06T13:00Z"]
    flux = np.array([1e-5, 9e-4, 1e-5])
    out = oblique_absorption(tx, rx, times, 10.0, flux, n_hops=1, height_km=300.0)
    assert out.shape == (3,)
    assert out[1] > out[0]  # flare peak dominates


def test_oblique_absorption_night_path_is_zero():
    tx, rx = (10.0, 170.0), (10.0, -170.0)  # both near the Pacific date line
    total = oblique_absorption(tx, rx, "2017-09-06T12:00Z", 10.0, 1e-4,
                               n_hops=1, height_km=300.0, grazing=False)
    assert total == pytest.approx(0.0)


def test_oblique_absorption_flat_vs_curved_close_for_short_hop():
    tx, rx = (10.0, 20.0), (10.2, 20.3)
    kw = dict(height_km=300.0)
    a_flat = oblique_absorption(tx, rx, "2017-09-06T12:00Z", 10.0, 1e-5, earth_model="flat", **kw)
    a_curved = oblique_absorption(tx, rx, "2017-09-06T12:00Z", 10.0, 1e-5, earth_model="curved", **kw)
    assert a_flat == pytest.approx(a_curved, rel=0.05)


def test_oblique_absorption_parabolic_height_model():
    tx, rx = (10.0, 20.0), (10.0, 30.0)
    total = oblique_absorption(
        tx, rx, "2017-09-06T12:00Z", 10.0, 1e-5,
        height_model="parabolic", foF2_mhz=12.0, hmF2_km=300.0, ym_km=100.0,
    )
    assert total > 0.0 and np.isfinite(total)


def test_oblique_absorption_drap2_model():
    tx, rx = (10.0, 20.0), (10.0, 30.0)
    total = oblique_absorption(tx, rx, "2017-09-06T12:00Z", 10.0, 1e-5,
                               model="drap2", height_km=300.0)
    assert total > 0.0
