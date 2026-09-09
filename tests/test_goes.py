"""Tests for xrap.goes.

Offline tests mock ``xrap.goes._get_json`` (the only HTTP touch-point for the
SWPC routes). Live tests are marked ``network`` and run only with
``pytest --run-network``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xrap.goes import _resolve_feed, _sat_number, fetch_goes_flares, fetch_goes_xrs


# --------------------------------------------------------------------------- #
# Argument validation (offline)
# --------------------------------------------------------------------------- #
def test_rejects_backwards_interval():
    with pytest.raises(ValueError):
        fetch_goes_xrs("2017-09-06T13:00", "2017-09-06T11:00")


def test_netcdf_url_requires_url():
    with pytest.raises(ValueError):
        fetch_goes_xrs("2005-01-01", "2005-01-02", source="netcdf_url")


def test_local_requires_path():
    with pytest.raises(ValueError):
        fetch_goes_xrs("2005-01-01", "2005-01-02", source="local")


def test_noaa_json_rejects_old_events():
    old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    with pytest.raises(ValueError, match="last ~7 days"):
        fetch_goes_xrs(old, old + pd.Timedelta(hours=1), source="noaa_json")


@pytest.mark.parametrize(
    "value, expected",
    [(16, 16), ("16", 16), ("GOES-16", 16), ("goes18", 18),
     (None, None), ("primary", None), ("secondary", None)],
)
def test_sat_number_parsing(value, expected):
    assert _sat_number(value) == expected


# --------------------------------------------------------------------------- #
# NOAA SWPC JSON route (mocked)
# --------------------------------------------------------------------------- #
def _fake_xrays_records(times, satellite=18):
    """Two rows per timestamp (one per band), SWPC ``xrays-*`` schema."""
    out = []
    for i, t in enumerate(times):
        stamp = pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({"time_tag": stamp, "satellite": satellite,
                    "flux": 1e-8 + i, "observed_flux": 2e-8 + i * 1e-9,
                    "energy": "0.05-0.4nm"})
        out.append({"time_tag": stamp, "satellite": satellite,
                    "flux": 3e-7 + i, "observed_flux": 4e-7 + i * 1e-8,
                    "energy": "0.1-0.8nm"})
    return out


def test_noaa_json_uses_observed_flux_for_both_bands(mocker):
    now = pd.Timestamp.now(tz="UTC").floor("min")
    times = pd.date_range(now - pd.Timedelta(minutes=4), now, freq="1min")
    mocker.patch("xrap.goes._get_json", return_value=_fake_xrays_records(times))

    ds = fetch_goes_xrs(now - pd.Timedelta(hours=2), now, source="noaa_json")

    assert set(ds.sizes) == {"time"}
    assert ds.sizes["time"] == len(times)
    assert ds.attrs == {**ds.attrs, "source": "noaa_json", "quality": "nrt",
                        "satellite": "GOES-18", "units": "W m-2"}
    # observed_flux, not flux
    np.testing.assert_allclose(ds["xrsa"].values[0], 2e-8)
    np.testing.assert_allclose(ds["xrsb"].values[0], 4e-7)


def test_noaa_json_selects_narrowest_feed(mocker):
    spy = mocker.patch("xrap.goes._get_json", return_value=_fake_xrays_records(
        pd.date_range(pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=2),
                      periods=3, freq="1min")))
    now = pd.Timestamp.now(tz="UTC")
    fetch_goes_xrs(now - pd.Timedelta(hours=3), now, source="noaa_json")
    assert spy.call_args[0][0].endswith("/primary/xrays-6-hour.json")
    fetch_goes_xrs(now - pd.Timedelta(days=2), now, source="noaa_json")
    assert spy.call_args[0][0].endswith("/primary/xrays-3-day.json")


@pytest.mark.parametrize(
    "value, fname",
    [("6-hour", "xrays-6-hour.json"), ("6h", "xrays-6-hour.json"),
     ("1-day", "xrays-1-day.json"), ("1d", "xrays-1-day.json"),
     ("3-day", "xrays-3-day.json"), ("7-day", "xrays-7-day.json"),
     ("xrays-3-day.json", "xrays-3-day.json")],
)
def test_resolve_feed(value, fname):
    assert _resolve_feed(value) == fname


def test_resolve_feed_rejects_unknown():
    with pytest.raises(ValueError, match="6-hour"):
        _resolve_feed("30-day")


def test_noaa_json_explicit_feed_overrides_autopick(mocker):
    recs = _fake_xrays_records(
        pd.date_range(pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=2),
                      periods=3, freq="1min"))
    spy = mocker.patch("xrap.goes._get_json", return_value=recs)
    # no start/end needed when feed= is given
    ds = fetch_goes_xrs(source="noaa_json", feed="6-hour")
    assert spy.call_args[0][0].endswith("/primary/xrays-6-hour.json")
    assert ds.sizes["time"] == 3


def test_noaa_json_secondary_spacecraft(mocker):
    spy = mocker.patch("xrap.goes._get_json", return_value=_fake_xrays_records(
        pd.date_range(pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=2),
                      periods=3, freq="1min")))
    now = pd.Timestamp.now(tz="UTC")
    fetch_goes_xrs(now - pd.Timedelta(hours=1), now, source="noaa_json", satellite="secondary")
    assert "/secondary/" in spy.call_args[0][0]


# --------------------------------------------------------------------------- #
# Flare-event feed (mocked)
# --------------------------------------------------------------------------- #
_FLARE_RECORDS = [
    {"time_tag": "2026-09-02T18:57:00Z", "begin_time": "2026-09-02T18:57:00Z",
     "max_time": "2026-09-02T19:20:00Z", "max_class": "M3.0",
     "max_xrlong": 3.0e-05, "end_time": "2026-09-02T19:48:00Z", "satellite": 18},
    {"time_tag": "2026-09-05T02:10:00Z", "begin_time": "2026-09-05T02:10:00Z",
     "max_time": "2026-09-05T02:31:00Z", "max_class": "C1.5",
     "max_xrlong": 1.5e-06, "end_time": "2026-09-05T02:44:00Z", "satellite": 18},
]


def test_fetch_flares_parses_times_and_sorts(mocker):
    mocker.patch("xrap.goes._get_json", return_value=_FLARE_RECORDS)
    df = fetch_goes_flares()
    assert list(df["max_class"]) == ["M3.0", "C1.5"]
    assert df["max_time"].dt.tz is not None
    assert df.index.name == "max_time"


def test_fetch_flares_time_window(mocker):
    mocker.patch("xrap.goes._get_json", return_value=_FLARE_RECORDS)
    df = fetch_goes_flares(start="2026-09-04", end="2026-09-06")
    assert list(df["max_class"]) == ["C1.5"]


def test_fetch_flares_latest_endpoint(mocker):
    spy = mocker.patch("xrap.goes._get_json", return_value=_FLARE_RECORDS[:1])
    fetch_goes_flares(latest=True)
    assert spy.call_args[0][0].endswith("xray-flares-latest.json")


# --------------------------------------------------------------------------- #
# Live (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_sunpy_route_live():
    ds = fetch_goes_xrs("2017-09-06T11:00", "2017-09-06T13:00",
                        source="sunpy", satellite=16)
    assert {"xrsa", "xrsb"} <= set(ds.data_vars)
    assert ds.attrs["quality"] == "science"
    # the 2017-09-06 X9.3 flare peaks ~12:02 UT
    peak = pd.Timestamp(ds["time"].values[int(np.argmax(ds["xrsb"].values))])
    assert abs((peak - pd.Timestamp("2017-09-06T12:02")).total_seconds()) < 300
    assert ds["xrsb"].max() > 1e-4


@pytest.mark.network
def test_noaa_json_route_live():
    now = pd.Timestamp.now(tz="UTC")
    ds = fetch_goes_xrs(now - pd.Timedelta(hours=3), now, source="noaa_json")
    assert ds.sizes["time"] > 0
    assert ds.attrs["quality"] == "nrt"
    assert np.nanmedian(ds["xrsb"].values) > 0
