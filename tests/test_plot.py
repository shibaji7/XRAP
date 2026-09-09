"""Tests for xrap.plot.

The offline test exercises the plotting code on the synthetic fixture. The two
``network`` tests download real GOES data (2017-09 flare via SunPy; the last
day via the SWPC feed), plot it, and save a PNG -- run with
``pytest --run-network``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt

from xrap import AbsorptionModel, fetch_goes_xrs, plot_absorption, plot_xrs


def test_plot_xrs_log_axis_and_traces(synthetic_xrs):
    ax = plot_xrs(synthetic_xrs)
    assert ax.get_yscale() == "log"
    labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert any("0.1-0.8" in lbl for lbl in labels)
    assert any("0.05-0.4" in lbl for lbl in labels)
    plt.close(ax.figure)


def test_plot_xrs_accepts_external_axis(synthetic_xrs):
    fig, ax = plt.subplots()
    assert plot_xrs(synthetic_xrs, ax=ax, flare_classes=False) is ax
    plt.close(fig)


def test_plot_xrs_saves_png(synthetic_xrs, tmp_path):
    ax = plot_xrs(synthetic_xrs, title="synthetic flare")
    out = tmp_path / "xrs_synthetic.png"
    ax.figure.savefig(out, dpi=80)
    plt.close(ax.figure)
    assert out.stat().st_size > 1000


@pytest.mark.network
def test_plot_2017_september_flare(tmp_path):
    """Download the 2017-09-06 X9.3 flare and plot flux (log) vs time."""
    ds = fetch_goes_xrs("2017-09-06T11:00", "2017-09-06T13:00",
                        source="sunpy", satellite=16)
    assert float(ds["xrsb"].max()) > 1e-4  # X-class

    ax = plot_xrs(ds, title="GOES-16 XRS -- 2017-09-06 X9.3")
    out = tmp_path / "xrs_2017-09-06.png"
    ax.figure.savefig(out, dpi=110)
    plt.close(ax.figure)
    assert out.stat().st_size > 5000

    peak = pd.Timestamp(ds["time"].values[int(np.argmax(ds["xrsb"].values))])
    assert abs((peak - pd.Timestamp("2017-09-06T12:02")).total_seconds()) < 300


@pytest.mark.network
def test_plot_last_1_day(tmp_path):
    """Download the last day of GOES XRS from the SWPC feed and plot it."""
    ds = fetch_goes_xrs(source="noaa_json", feed="1-day")
    assert ds.sizes["time"] > 500              # ~1440 min in a day
    assert np.nanmedian(ds["xrsb"].values) > 0

    ax = plot_xrs(ds, title="GOES XRS -- last 24 h")
    out = tmp_path / "xrs_last_1day.png"
    ax.figure.savefig(out, dpi=110)
    plt.close(ax.figure)
    assert out.stat().st_size > 5000


# --------------------------------------------------------------------------- #
# End-to-end: fetch GOES -> absorption at 10 MHz for BOTH models -> overlay plot
# --------------------------------------------------------------------------- #
def _predict_both(ds, *, lat, lon, freq_mhz=10.0):
    """Run the 'xrap' and 'drap2' models on one GOES dataset; return {name: df}."""
    out = {}
    for name in ("xrap", "drap2"):
        m = AbsorptionModel(freq_mhz=freq_mhz, model=name, sza_method="noaa")
        out[name] = m.predict(lat=lat, lon=lon,
                              start=ds["time"].values[0], end=ds["time"].values[-1],
                              xray=ds)
    return out


def _overlay(dfs, *, title, out_path, freq_label="10 MHz"):
    colors = {"xrap": "#2ca02c", "drap2": "#1f77b4"}
    ax = None
    for name, df in dfs.items():
        ax = plot_absorption(df, ax=ax, label=name.upper(), color=colors[name],
                             freq_label=freq_label, title=title)
    ax.figure.savefig(out_path, dpi=110)
    plt.close(ax.figure)


@pytest.mark.network
def test_absorption_2017_flare_10mhz_both_models(tmp_path):
    """2017-09-06 X9.3: 10 MHz absorption, XRAP vs DRAP2 overlaid."""
    ds = fetch_goes_xrs("2017-09-06T09:00", "2017-09-06T15:00",
                        source="sunpy", satellite=16)
    dfs = _predict_both(ds, lat=20.0, lon=30.0)

    for name, df in dfs.items():
        assert (df["sza_deg"] < 90).any()
        assert df["absorption_db"].max() > 1.0, name
        off = abs((df["absorption_db"].idxmax() - df["xrsb"].idxmax()).total_seconds())
        assert off <= 180, name
    # XRAP (Fiori 2022) sits well above DRAP2 for a strong flare
    assert dfs["xrap"]["absorption_db"].max() > dfs["drap2"]["absorption_db"].max()

    out = tmp_path / "abs_2017-09-06_10MHz_xrap_vs_drap2.png"
    _overlay(dfs, title="10 MHz absorption -- 2017-09-06 X9.3, 20N 30E", out_path=out)
    assert out.stat().st_size > 5000


@pytest.mark.network
def test_absorption_last_1_day_10mhz_both_models(tmp_path):
    """Last 24 h: 10 MHz absorption at the sub-solar meridian, XRAP vs DRAP2 overlaid."""
    ds = fetch_goes_xrs(source="noaa_json", feed="1-day")
    dfs = _predict_both(ds, lat=0.0, lon=0.0)

    for name, df in dfs.items():
        assert (df["absorption_db"] == 0).any(), name   # night side -> zero
        assert (df["absorption_db"] > 0).any(), name    # dayside -> non-zero

    out = tmp_path / "abs_last_1day_10MHz_xrap_vs_drap2.png"
    _overlay(dfs, title="10 MHz absorption -- last 24 h, 0N 0E", out_path=out)
    assert out.stat().st_size > 5000
