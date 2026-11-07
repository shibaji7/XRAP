"""Tests for xrap.absorption -- the "xrap" (Fiori et al. 2022) and "drap2" models."""

from __future__ import annotations

import numpy as np
import pytest

from xrap import absorption, peak_scale, reference_absorption, scale_frequency
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


# --------------------------------------------------------------------------- #
# max_db blackout cap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["xrap", "drap2"])
def test_max_db_caps_vertical_absorption(model):
    raw = absorption(3.75, 20.0, 9e-4, model=model)
    capped = absorption(3.75, 20.0, 9e-4, model=model, max_db=5.0)
    assert raw > 5.0
    assert capped == pytest.approx(5.0)
    # below the cap: untouched
    assert absorption(30.0, 20.0, 1e-6, model=model, max_db=5.0) == pytest.approx(
        absorption(30.0, 20.0, 1e-6, model=model))


# --------------------------------------------------------------------------- #
# Low-frequency saturation of the f^-n law
# --------------------------------------------------------------------------- #
def test_scale_frequency_saturation_is_continuous_at_the_corner():
    lo = scale_frequency(1.0, 30.0, 5.0 - 1e-9, f_sat_mhz=5.0)
    hi = scale_frequency(1.0, 30.0, 5.0, f_sat_mhz=5.0)
    assert lo == pytest.approx(hi, rel=1e-6)
    assert hi == pytest.approx((30.0 / 5.0) ** 1.5)


def test_scale_frequency_saturation_leaves_the_power_law_above_the_corner():
    f = np.array([5.0, 7.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(scale_frequency(1.0, 30.0, f, f_sat_mhz=5.0),
                               scale_frequency(1.0, 30.0, f))


def test_scale_frequency_saturation_is_bounded_below_the_corner():
    f = np.array([4.0, 3.75, 1.8, 0.5, 0.01, 1e-6])
    s = scale_frequency(1.0, 30.0, f, f_sat_mhz=5.0)
    limit = 2.0 * (30.0 / 5.0) ** 1.5
    assert np.all(s < limit)
    assert np.all(np.diff(s) > 0)                      # still decreasing with f, i.e. rising to lower f
    assert s[-1] == pytest.approx(limit, rel=1e-4)
    # the unsaturated law diverges there
    assert scale_frequency(1.0, 30.0, 1e-6) > 1e6 * limit


def test_scale_frequency_saturation_respects_a_custom_exponent():
    s = scale_frequency(1.0, 30.0, 1e-9, n=2.0, f_sat_mhz=4.0)
    assert s == pytest.approx(2.0 * (30.0 / 4.0) ** 2.0, rel=1e-6)


def test_xrap_saturates_below_5mhz_by_default_and_not_above():
    # >= 5 MHz identical to the unsaturated law
    for f in (5.0, 10.0, 30.0):
        assert absorption(f, 0.0, 9e-4, grazing=False) == pytest.approx(
            absorption(f, 0.0, 9e-4, grazing=False, freq_saturation_mhz=None))
    # below 5 MHz smaller, and bounded
    sat = absorption(3.75, 0.0, 9e-4, grazing=False)
    raw = absorption(3.75, 0.0, 9e-4, grazing=False, freq_saturation_mhz=None)
    assert sat < raw
    assert sat == pytest.approx(12080.0 * 9e-4 * 2 * 30**1.5 / (3.75**1.5 + 5.0**1.5))
    assert absorption(1e-6, 0.0, 9e-4, grazing=False) == pytest.approx(
        12080.0 * 9e-4 * 2.0 * (30.0 / 5.0) ** 1.5, rel=1e-4)


def test_saturation_corner_is_user_selectable():
    # a higher corner saturates earlier -> a lower ceiling 2*(30/fs)^1.5 -> smaller value
    a3 = absorption(2.0, 0.0, 9e-4, grazing=False, freq_saturation_mhz=3.0)
    a8 = absorption(2.0, 0.0, 9e-4, grazing=False, freq_saturation_mhz=8.0)
    assert a8 < a3 < absorption(2.0, 0.0, 9e-4, grazing=False, freq_saturation_mhz=None)


def test_saturation_can_be_set_through_coeffs():
    c = load_coefficients("xrap")
    c["freq_saturation_mhz"] = None
    assert absorption(3.75, 0.0, 9e-4, grazing=False, coeffs=c) == pytest.approx(
        absorption(3.75, 0.0, 9e-4, grazing=False, freq_saturation_mhz=None))


def test_drap2_is_unsaturated_by_default_but_can_opt_in():
    assert load_coefficients("drap2").get("freq_saturation_mhz") is None
    default = absorption(3.75, 0.0, 9e-4, model="drap2")
    assert default == pytest.approx(absorption(3.75, 0.0, 9e-4, model="drap2",
                                               freq_saturation_mhz=None))
    opt_in = absorption(3.75, 0.0, 9e-4, model="drap2", freq_saturation_mhz=5.0)
    assert opt_in < default
    # unchanged at/above the corner
    assert absorption(10.0, 0.0, 9e-4, model="drap2", freq_saturation_mhz=5.0) == pytest.approx(
        absorption(10.0, 0.0, 9e-4, model="drap2"))


def test_invalid_saturation_string_is_rejected():
    with pytest.raises(ValueError):
        absorption(3.75, 0.0, 9e-4, freq_saturation_mhz="banana")


# --------------------------------------------------------------------------- #
# max_db: peak scaling of the whole series (default) vs hard clip
# --------------------------------------------------------------------------- #
def test_peak_scale_puts_the_peak_on_max_db_and_keeps_the_shape():
    a = np.array([2.0, 10.0, 383.0, 73.0, 4.0])
    s, k = peak_scale(a, 100.0)
    assert s.max() == pytest.approx(100.0)
    assert k == pytest.approx(100.0 / 383.0)
    np.testing.assert_allclose(s / a, k)                 # one factor for every element
    assert np.argmax(s) == np.argmax(a)
    assert np.sum(s == 100.0) == 1                       # no flat top


def test_peak_scale_never_scales_up():
    a = np.array([1.0, 50.0, 3.0])
    s, k = peak_scale(a, 100.0)
    assert k == 1.0
    np.testing.assert_array_equal(s, a)


@pytest.mark.parametrize("bad", [np.array([np.nan, np.nan]), np.array([0.0, 0.0]), np.array([])])
def test_peak_scale_leaves_degenerate_input_alone(bad):
    s, k = peak_scale(bad, 10.0)
    assert k == 1.0
    np.testing.assert_array_equal(s, bad)


def test_peak_scale_ignores_nans_when_finding_the_peak():
    s, k = peak_scale(np.array([np.nan, 200.0, 20.0]), 100.0)
    assert k == pytest.approx(0.5)
    assert s[1] == pytest.approx(100.0) and s[2] == pytest.approx(10.0)


def test_max_db_scales_an_absorption_time_series():
    flux = np.array([1e-6, 9e-4, 1e-5])
    raw = absorption(3.75, 20.0, flux)
    sc = absorption(3.75, 20.0, flux, max_db=50.0)
    assert raw.max() > 50.0
    assert sc.max() == pytest.approx(50.0)
    np.testing.assert_allclose(sc / raw, 50.0 / raw.max())   # shape preserved
    assert sc[0] < raw[0]                                    # quiet values are rescaled too


def test_max_db_clip_mode_is_the_old_hard_cut():
    flux = np.array([1e-6, 9e-4, 1e-5])
    raw = absorption(3.75, 20.0, flux)
    cl = absorption(3.75, 20.0, flux, max_db=50.0, max_db_mode="clip")
    np.testing.assert_allclose(cl, np.minimum(raw, 50.0))


def test_max_db_below_the_peak_is_a_no_op_in_scale_mode():
    flux = np.array([1e-6, 1e-5])
    raw = absorption(30.0, 20.0, flux)
    np.testing.assert_allclose(absorption(30.0, 20.0, flux, max_db=1e6), raw)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_max_db_must_be_positive(bad):
    with pytest.raises(ValueError):
        absorption(10.0, 20.0, 1e-4, max_db=bad)


def test_max_db_mode_rejects_unknown_values():
    with pytest.raises(ValueError):
        absorption(10.0, 20.0, 1e-4, max_db=5.0, max_db_mode="banana")
