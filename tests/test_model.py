"""Tests for xrap.model.AbsorptionModel (end-to-end wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xrap.model import AbsorptionModel


def test_construct_with_defaults():
    m = AbsorptionModel()
    assert m.freq_mhz == 10.0
    assert m.model == "xrap"
    assert m.sza_method == "noaa"
    assert m.coeffs is None  # resolved per-model at call time


def test_drap2_model_selectable():
    m = AbsorptionModel(model="drap2")
    val = m.predict_point(20.0, 30.0, "2017-09-06T12:00Z", xray_wm2=1e-4)
    assert val > 0.0


def test_predict_with_prefetched_xray(synthetic_xrs):
    """predict() should not touch the network when xray= is supplied.

    Dayside location (lon ~ 30 E) so the 2017-09-06 ~12 UT flare is in sunlight.
    """
    m = AbsorptionModel(freq_mhz=10.0)
    df = m.predict(
        lat=10.0, lon=30.0,
        start=synthetic_xrs.time.values[0],
        end=synthetic_xrs.time.values[-1],
        xray=synthetic_xrs,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["sza_deg", "xrsb", "absorption_db"]
    assert len(df) == synthetic_xrs.sizes["time"]
    assert np.all(df["sza_deg"] < 90.0)                       # dayside
    assert df["absorption_db"].max() > 0.0
    # absorption tracks the X-ray flux: its peak sits within a couple of
    # minutes of the flux peak (small offset from the slow SZA trend).
    offset = abs((df["absorption_db"].idxmax() - df["xrsb"].idxmax()).total_seconds())
    assert offset <= 120.0


def test_predict_point():
    m = AbsorptionModel()
    val = m.predict_point(10.0, 30.0, "2017-09-06T12:00Z", xray_wm2=1e-4)
    assert val > 0.0


def test_predict_point_night_is_zero():
    m = AbsorptionModel()
    # local midnight at lon 30 E -> ~22 UT; use 00 UT (2 am local) -> dark
    assert m.predict_point(10.0, 30.0, "2017-09-06T00:00Z", xray_wm2=1e-4) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# max_db on AbsorptionModel
# --------------------------------------------------------------------------- #
def _predict(synthetic_xrs, **kw):
    m = AbsorptionModel(freq_mhz=3.75, **kw)
    return m.predict(lat=10.0, lon=30.0, start=synthetic_xrs.time.values[0],
                     end=synthetic_xrs.time.values[-1], xray=synthetic_xrs)


def test_default_columns_are_unchanged_without_max_db(synthetic_xrs):
    assert list(_predict(synthetic_xrs).columns) == ["sza_deg", "xrsb", "absorption_db"]


def test_max_db_scales_the_predicted_series_and_keeps_the_raw_values(synthetic_xrs):
    raw = _predict(synthetic_xrs)["absorption_db"]
    df = _predict(synthetic_xrs, max_db=50.0)
    assert list(df.columns) == ["sza_deg", "xrsb", "absorption_db", "absorption_raw_db"]
    assert raw.max() > 50.0
    assert df["absorption_db"].max() == pytest.approx(50.0)
    np.testing.assert_allclose(df["absorption_raw_db"], raw)            # untouched
    np.testing.assert_allclose(df["absorption_db"], raw * (50.0 / raw.max()))   # shape kept
    assert df["absorption_db"].idxmax() == raw.idxmax()


def test_max_db_clip_mode_flattens_the_peak(synthetic_xrs):
    raw = _predict(synthetic_xrs)["absorption_db"]
    df = _predict(synthetic_xrs, max_db=50.0, max_db_mode="clip")
    np.testing.assert_allclose(df["absorption_db"], np.minimum(raw, 50.0))


def test_max_db_above_the_peak_changes_nothing(synthetic_xrs):
    raw = _predict(synthetic_xrs)["absorption_db"]
    np.testing.assert_allclose(_predict(synthetic_xrs, max_db=1e6)["absorption_db"], raw)


def test_predict_point_respects_max_db():
    m = AbsorptionModel(freq_mhz=3.75, max_db=20.0)
    assert m.predict_point(10.0, 30.0, "2017-09-06T12:00Z", xray_wm2=9e-4) == pytest.approx(20.0)
    assert AbsorptionModel(freq_mhz=3.75).predict_point(
        10.0, 30.0, "2017-09-06T12:00Z", xray_wm2=9e-4) > 20.0
