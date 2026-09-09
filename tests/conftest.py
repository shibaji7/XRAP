"""Shared pytest fixtures and configuration for the XRAP test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run tests marked @pytest.mark.network (hit GOES/SWPC)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test reaches external services (GOES archive / SWPC)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_times():
    """A short 1-min index around the 2017-09-06 X9.3 flare.

    Timezone-naive UTC datetime64, matching what :func:`xrap.fetch_goes_xrs`
    returns.
    """
    return pd.date_range("2017-09-06T11:30", "2017-09-06T12:30", freq="1min")


@pytest.fixture
def synthetic_xrs(sample_times):
    """A fake GOES/XRS dataset (Gaussian flare on a quiet background).

    Lets the absorption / model tests run without any network access.
    """
    t = np.arange(len(sample_times))
    peak = len(t) // 2
    xrsb = 1e-8 + 9e-4 * np.exp(-0.5 * ((t - peak) / 6.0) ** 2)  # ~X9 peak
    xrsa = 0.1 * xrsb
    return xr.Dataset(
        {
            "xrsa": ("time", xrsa),
            "xrsb": ("time", xrsb),
        },
        coords={"time": sample_times},
        attrs={"satellite": "GOES-16", "source": "synthetic", "quality": "test"},
    )


@pytest.fixture
def xrap_coeffs():
    from xrap.absorption import load_coefficients

    return load_coefficients("xrap")


@pytest.fixture
def drap2_coeffs():
    from xrap.absorption import load_coefficients

    return load_coefficients("drap2")
