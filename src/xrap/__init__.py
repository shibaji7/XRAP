"""XRAP -- X-Ray Absorption Prediction.

Estimate solar-flare-driven HF radio-wave absorption for a specific HF
frequency, parameterized by:

* solar zenith angle (SZA) at the absorbing point,
* HF frequency,
* solar soft X-ray flux (GOES/XRS, typically the 0.1-0.8 nm long band).

Typical entry points
--------------------
>>> import xrap
>>> chi = xrap.solar_zenith_angle(lat=40.0, lon=-105.0, time="2017-09-06T12:02Z")
>>> xrs = xrap.fetch_goes_xrs("2017-09-06T11:00", "2017-09-06T13:00")
>>> dB = xrap.absorption(freq_mhz=10.0, sza_deg=chi, xray_wm2=xrs["xrsb"])  # model="xrap"

Submodules
----------
sza         -- solar geometry helpers (SZA from lat/lon/time)
goes        -- GOES/XRS X-ray flux retrieval (SunPy for >= 2010, fallbacks for prior events)
absorption  -- absorption models: "xrap" (Fiori et al. 2022) and "drap2"
model       -- high-level orchestration tying the pieces together
cli         -- command-line interface
"""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - only during a source checkout without build
    __version__ = "0.0.0.dev0"

from .absorption import (
    absorption,
    highest_affected_frequency,
    reference_absorption,
    scale_frequency,
)
from .goes import fetch_goes_flares, fetch_goes_xrs
from .model import AbsorptionModel
from .plot import plot_absorption, plot_xrs
from .sza import (
    chapman_function,
    is_sunlit,
    solar_zenith_angle,
    terminator_sza,
)

__all__ = [
    "__version__",
    "absorption",
    "reference_absorption",
    "highest_affected_frequency",
    "scale_frequency",
    "fetch_goes_xrs",
    "fetch_goes_flares",
    "AbsorptionModel",
    "plot_xrs",
    "plot_absorption",
    "solar_zenith_angle",
    "chapman_function",
    "is_sunlit",
    "terminator_sza",
]
