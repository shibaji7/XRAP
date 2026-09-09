"""Tests for xrap.absorption -- the "xrap" (Fiori et al. 2022) and "drap2" models."""

from __future__ import annotations

import numpy as np
import pytest

from xrap import absorption, reference_absorption, scale_frequency
from xrap.absorption import highest_affected_frequency, load_coefficients, zenith_factor


# --------------------------------------------------------------------------- #
# Coefficient loading
# --------------------------------------------------------------------------- #
def test_coefficient_sets():
    x = load_coefficients("xrap")
    assert x["scale"] == 12080.0
    assert x["f0_mhz"] == 30.0 and x["freq_exponent"] == 1.5
    d = load_coefficients("drap2")
    assert (d["haf_slope"], d["haf_intercept"]) == (10.0, 65.0)
    assert not any(k.startswith("_") for k in x) and not any(k.startswith("_") for k in d)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        load_coefficients("banana")
    with pytest.raises(ValueError):
        absorption(10.0, 30.0, 1e-4, model="banana")


# --------------------------------------------------------------------------- #
# XRAP model (default)
# --------------------------------------------------------------------------- #
def test_xrap_is_the_default_model():
    assert absorption(30.0, 0.0, 1e-4) == absorption(30.0, 0.0, 1e-4, model="xrap")


def test_xrap_a30_base_equation():
    """A(30) = 12080 * flux * cos(chi), one-way, at chi = 0 -> 12080 * flux."""
    a = absorption(30.0, 0.0, 1e-4, model="xrap", grazing=False)
    assert a == pytest.approx(12080.0 * 1e-4)  # 1.208 dB


def test_xrap_cos_sza_dependence():
    a0 = absorption(30.0, 0.0, 1e-4, grazing=False)
    a60 = absorption(30.0, 60.0, 1e-4, grazing=False)
    assert a60 == pytest.approx(a0 * np.cos(np.radians(60.0)))


def test_xrap_linear_in_flux():
    f = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    a = absorption(30.0, 20.0, f, grazing=False)
    np.testing.assert_allclose(a / f, a[0] / f[0])  # strictly proportional


def test_xrap_frequency_scaling_from_30mhz():
    a30 = absorption(30.0, 10.0, 1e-4, grazing=False)
    a10 = absorption(10.0, 10.0, 1e-4, grazing=False)
    assert a10 == pytest.approx(a30 * (30.0 / 10.0) ** 1.5)


def test_xrap_reference_absorption_matches():
    a30 = reference_absorption(15.0, 1e-4, grazing=False)
    assert absorption(30.0, 15.0, 1e-4, grazing=False) == pytest.approx(a30)
    # and scaling A(f0) reproduces absorption(f)
    assert scale_frequency(a30, 30.0, 12.0) == pytest.approx(
        absorption(12.0, 15.0, 1e-4, grazing=False)
    )


def test_xrap_hard_cutoff_at_90_when_grazing_off():
    assert absorption(10.0, 90.0, 1e-4, grazing=False) == pytest.approx(0.0)
    assert absorption(10.0, 95.0, 1e-4, grazing=False) == pytest.approx(0.0)


def test_xrap_grazing_default_gives_small_positive_past_90():
    # grazing defaults to True for xrap
    assert absorption(10.0, 95.0, 1e-4, alt_km=90.0) > 0.0
    assert absorption(10.0, 95.0, 1e-4) < absorption(10.0, 85.0, 1e-4)
    # zero beyond the 90 km terminator (~99.6 deg)
    assert absorption(10.0, 105.0, 1e-4) == pytest.approx(0.0)


def test_xrap_grazing_equals_cos_away_from_terminator():
    kw = dict(freq_mhz=10.0, sza_deg=45.0, xray_wm2=1e-4)
    assert absorption(**kw, grazing=True) == pytest.approx(absorption(**kw, grazing=False), rel=0.02)


def test_xrap_zero_flux_and_negative_clip():
    assert absorption(10.0, 30.0, 0.0) == pytest.approx(0.0)
    assert absorption(10.0, 30.0, -1.0) == pytest.approx(0.0)


def test_xrap_custom_exponents_via_coeffs():
    c = load_coefficients("xrap")
    c["flux_exponent"] = 0.5
    a = absorption(30.0, 0.0, 1e-4, model="xrap", coeffs=c, grazing=False)
    assert a == pytest.approx(12080.0 * (1e-4**0.5))


# --------------------------------------------------------------------------- #
# path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["xrap", "drap2"])
def test_total_is_twice_oneway(model):
    kw = dict(freq_mhz=10.0, sza_deg=25.0, xray_wm2=1e-4, model=model)
    assert absorption(**kw, path="total") == pytest.approx(2.0 * absorption(**kw, path="oneway"))


def test_invalid_path_raises():
    with pytest.raises(ValueError):
        absorption(10.0, 30.0, 1e-4, path="round-trip")


# --------------------------------------------------------------------------- #
# DRAP2 model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flux, haf0", [(1e-5, 15.0), (1e-4, 25.0), (1e-3, 35.0)])
def test_drap2_haf_subsolar_benchmarks(flux, haf0):
    assert highest_affected_frequency(flux, 0.0) == pytest.approx(haf0)


def test_drap2_haf_zero_night_and_low_flux():
    assert highest_affected_frequency(1e-4, 90.0) == pytest.approx(0.0)
    assert highest_affected_frequency(1e-8, 0.0) == pytest.approx(0.0)


def test_drap2_absorption_equals_half_threshold_at_haf():
    """One-way A(f = HAF) = 0.5 * haf_threshold_db; total = 1 dB (the definition)."""
    haf = float(highest_affected_frequency(1e-4, 30.0))
    assert absorption(haf, 30.0, 1e-4, model="drap2", path="oneway") == pytest.approx(0.5)
    assert absorption(haf, 30.0, 1e-4, model="drap2", path="total") == pytest.approx(1.0)


def test_drap2_frequency_scaling():
    a1 = absorption(10.0, 20.0, 1e-4, model="drap2")
    a2 = absorption(20.0, 20.0, 1e-4, model="drap2")
    assert a2 / a1 == pytest.approx(0.5**1.5)


def test_drap2_hard_cutoff_is_default():
    assert absorption(10.0, 95.0, 1e-4, model="drap2") == pytest.approx(0.0)
    assert absorption(10.0, 95.0, 1e-4, model="drap2", grazing=True, alt_km=90.0) > 0.0


def test_drap2_monotonic_in_flux_and_broadcast():
    flux = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    a = absorption(10.0, 30.0, flux, model="drap2")
    assert np.all(np.diff(a) > 0)
    f = np.array([5.0, 10.0])[:, None]
    phi = np.array([1e-5, 1e-4, 1e-3])[None, :]
    assert absorption(f, 20.0, phi, model="drap2").shape == (2, 3)


# --------------------------------------------------------------------------- #
# scale_frequency
# --------------------------------------------------------------------------- #
def test_scale_frequency_default_and_custom_exponent():
    assert scale_frequency(2.0, 10.0, 20.0) == pytest.approx(2.0 * 0.5**1.5)
    assert scale_frequency(2.0, 10.0, 10.0) == pytest.approx(2.0)
    assert scale_frequency(2.0, 10.0, 5.0, n=2.0) == pytest.approx(8.0)
    assert scale_frequency(1.0, 10.0, 20.0, n=0.0) == pytest.approx(1.0)


def test_scale_frequency_broadcasts():
    a = scale_frequency(np.array([1.0, 2.0, 4.0]), 10.0, 20.0)
    np.testing.assert_allclose(a, np.array([1.0, 2.0, 4.0]) * 0.5**1.5)


# --------------------------------------------------------------------------- #
# zenith_factor (shared primitive)
# --------------------------------------------------------------------------- #
def test_zenith_factor_hard():
    assert zenith_factor(0.0) == pytest.approx(1.0)
    assert zenith_factor(60.0) == pytest.approx(0.5)
    assert zenith_factor(90.0) == pytest.approx(0.0)
    assert zenith_factor(120.0) == pytest.approx(0.0)


def test_zenith_factor_grazing_small_positive_at_90():
    v90 = float(zenith_factor(90.0, grazing=True, alt_km=90.0))
    assert 0.0 < v90 < 0.05
    assert zenith_factor(95.0, grazing=True, alt_km=90.0) > 0.0
    assert zenith_factor(105.0, grazing=True, alt_km=90.0) == pytest.approx(0.0)
    # matches cos(chi) well away from the terminator
    assert zenith_factor(40.0, grazing=True) == pytest.approx(np.cos(np.radians(40.0)), rel=0.03)


# --------------------------------------------------------------------------- #
# polar-cap stub
# --------------------------------------------------------------------------- #
def test_polar_cap_absorption_not_implemented():
    from xrap.absorption import polar_cap_absorption

    with pytest.raises(NotImplementedError):
        polar_cap_absorption()
