"""Tests for the ``xrap`` command-line interface (offline)."""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from xrap.cli import build_parser, main


def _fake_xrays_records(times, satellite=18):
    """Two rows per timestamp (one per band), SWPC ``xrays-*`` schema."""
    out = []
    for i, t in enumerate(times):
        stamp = pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({"time_tag": stamp, "satellite": satellite,
                    "flux": 1e-8, "observed_flux": 2e-8, "energy": "0.05-0.4nm"})
        out.append({"time_tag": stamp, "satellite": satellite,
                    "flux": 3e-7, "observed_flux": 4e-7, "energy": "0.1-0.8nm"})
    return out


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_predict_writes_csv(tmp_path, synthetic_xrs, mocker):
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)
    out = tmp_path / "abs.csv"
    rc = main([
        "predict", "--lat", "20", "--lon", "30", "--freq", "10",
        "--start", "2017-09-06T11:30", "--end", "2017-09-06T12:30",
        "--sza-method", "noaa", "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out, index_col=0, parse_dates=True)
    assert list(df.columns) == ["sza_deg", "xrsb", "absorption_db"]
    assert len(df) == synthetic_xrs.sizes["time"]
    assert df["absorption_db"].max() > 0.0


def test_predict_saves_plot(tmp_path, synthetic_xrs, mocker):
    pytest.importorskip("matplotlib")
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)
    png = tmp_path / "abs.png"
    rc = main([
        "predict", "--lat", "20", "--lon", "30",
        "--start", "2017-09-06T11:30", "--end", "2017-09-06T12:30",
        "--plot", str(png), "-o", str(tmp_path / "abs.csv"),
    ])
    assert rc == 0
    assert png.stat().st_size > 1000


def test_predict_total_path_doubles_oneway(tmp_path, synthetic_xrs, mocker):
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)

    def run(path):
        out = tmp_path / f"{path}.csv"
        main(["predict", "--lat", "20", "--lon", "30",
              "--start", "2017-09-06T11:30", "--end", "2017-09-06T12:30",
              "--path", path, "-o", str(out)])
        return pd.read_csv(out, index_col=0)["absorption_db"].max()

    assert run("total") == pytest.approx(2.0 * run("oneway"), rel=1e-6)


def test_fetch_goes_writes_csv(tmp_path, mocker):
    now = pd.Timestamp.now(tz="UTC").floor("min")
    times = pd.date_range(now - pd.Timedelta(minutes=5), now, freq="1min")
    mocker.patch("xrap.goes._get_json", return_value=_fake_xrays_records(times))
    out = tmp_path / "xrs.csv"
    rc = main(["fetch-goes", "--source", "noaa_json", "--feed", "6-hour", "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert {"xrsa", "xrsb"} <= set(df.columns)


def test_fetch_goes_writes_netcdf(tmp_path, mocker):
    pytest.importorskip("netCDF4")
    now = pd.Timestamp.now(tz="UTC").floor("min")
    times = pd.date_range(now - pd.Timedelta(minutes=5), now, freq="1min")
    mocker.patch("xrap.goes._get_json", return_value=_fake_xrays_records(times))
    out = tmp_path / "xrs.nc"
    rc = main(["fetch-goes", "--source", "noaa_json", "--feed", "1-day", "-o", str(out)])
    assert rc == 0
    import xarray as xr

    with xr.open_dataset(out) as ds:
        assert {"xrsa", "xrsb"} <= set(ds.data_vars)


def test_freq_saturation_flag_parsing():
    from xrap.cli import _saturation_arg

    assert _saturation_arg("none") is None
    assert _saturation_arg("default") == "default"
    assert _saturation_arg("3.5") == 3.5
    with pytest.raises(argparse.ArgumentTypeError):
        _saturation_arg("banana")


def test_predict_freq_saturation_changes_low_frequency_output(tmp_path, synthetic_xrs, mocker):
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)

    def peak(extra):
        out = tmp_path / "o.csv"
        main(["predict", "--lat", "20", "--lon", "30", "--freq", "3.75",
              "--start", "2017-09-06T11:30", "--end", "2017-09-06T12:30",
              "-o", str(out), *extra])
        return pd.read_csv(out, index_col=0)["absorption_db"].max()

    assert peak(["--freq-saturation", "none"]) > peak([])              # default 5 MHz saturates
    assert peak(["--freq-saturation", "5"]) == pytest.approx(peak([]))


# --------------------------------------------------------------------------- #
# --max-db / --max-db-mode
# --------------------------------------------------------------------------- #
def _run_predict(tmp_path, extra):
    out = tmp_path / "o.csv"
    main(["predict", "--lat", "10", "--lon", "30", "--freq", "3.75",
          "--start", "2017-09-06T11:30", "--end", "2017-09-06T12:30",
          "-o", str(out), *extra])
    return pd.read_csv(out, index_col=0)


def test_predict_max_db_scales_the_series_and_writes_the_raw_column(tmp_path, synthetic_xrs, mocker):
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)
    base = _run_predict(tmp_path, [])
    df = _run_predict(tmp_path, ["--max-db", "40"])
    assert "absorption_raw_db" not in base.columns
    assert base["absorption_db"].max() > 40.0
    assert df["absorption_db"].max() == pytest.approx(40.0)
    assert df["absorption_raw_db"].max() == pytest.approx(base["absorption_db"].max())
    # shape preserved: one factor for every row
    ratio = (df["absorption_db"] / df["absorption_raw_db"]).dropna()
    assert ratio.max() == pytest.approx(ratio.min())


def test_predict_max_db_clip_mode(tmp_path, synthetic_xrs, mocker):
    mocker.patch("xrap.model.fetch_goes_xrs", return_value=synthetic_xrs)
    df = _run_predict(tmp_path, ["--max-db", "40", "--max-db-mode", "clip"])
    assert df["absorption_db"].max() == pytest.approx(40.0)
    assert (df["absorption_db"] == 40.0).sum() > 1            # flat top


@pytest.mark.parametrize("bad", ["0", "-5", "banana"])
def test_predict_rejects_a_non_positive_max_db(bad):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "--lat", "0", "--lon", "0",
                                   "--start", "a", "--end", "b", "--max-db", bad])
