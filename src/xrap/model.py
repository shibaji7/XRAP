"""High-level orchestration.

:class:`AbsorptionModel` ties the three pieces together so a user can go from
"a location, a time window, and (optionally) a frequency" straight to an
absorption time series, without hand-wiring the SZA and GOES steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .absorption import absorption
from .goes import fetch_goes_xrs
from .sza import solar_zenith_angle

__all__ = ["AbsorptionModel"]


@dataclass
class AbsorptionModel:
    """Configurable end-to-end absorption estimator.

    Parameters
    ----------
    freq_mhz : float
        Default HF frequency for :meth:`predict`.
    model : {"xrap", "drap2"}
        Absorption model (see :func:`xrap.absorption`).
    sza_method : str
        Passed to :func:`xrap.sza.solar_zenith_angle`.
    goes_source : str
        Passed to :func:`xrap.goes.fetch_goes_xrs`.
    path : {"oneway", "total"}
        Round-trip convention for :func:`xrap.absorption`.
    grazing : bool, optional
        Chapman grazing correction near the terminator; ``None`` -> per-model
        default (True for ``"xrap"``, False for ``"drap2"``).
    alt_km, H_km : float
        Absorbing-layer altitude and neutral scale height (grazing only).
    coeffs : dict, optional
        Coefficient override; defaults to the bundled set for ``model``.
    """

    freq_mhz: float = 10.0
    model: str = "xrap"
    sza_method: str = "noaa"
    goes_source: str = "sunpy"
    path: str = "oneway"
    grazing: bool | None = None
    alt_km: float = 90.0
    H_km: float = 7.0
    coeffs: dict | None = None

    def _absorption(self, freq_mhz, sza_deg, xray_wm2):
        return absorption(
            freq_mhz, sza_deg, xray_wm2,
            model=self.model, coeffs=self.coeffs, path=self.path,
            grazing=self.grazing, alt_km=self.alt_km, H_km=self.H_km,
        )

    def predict(
        self,
        lat: float,
        lon: float,
        start,
        end,
        *,
        freq_mhz: float | None = None,
        xray: xr.Dataset | None = None,
        **goes_kwargs,
    ) -> pd.DataFrame:
        """Absorption time series at ``(lat, lon)`` over ``[start, end]``.

        Parameters
        ----------
        lat, lon : float
            Absorbing-point geographic coordinates (deg, East-positive lon).
        start, end : str or datetime
            UTC window.
        freq_mhz : float, optional
            Override the instance default.
        xray : xarray.Dataset, optional
            Pre-fetched GOES/XRS data (skip the download).
        **goes_kwargs
            Forwarded to :func:`xrap.goes.fetch_goes_xrs`.

        Returns
        -------
        pandas.DataFrame
            Indexed by UTC time, columns ``sza_deg``, ``xrsb``, ``absorption_db``.
        """
        f = freq_mhz if freq_mhz is not None else self.freq_mhz

        if xray is None:
            xray = fetch_goes_xrs(start, end, source=self.goes_source, **goes_kwargs)

        times = pd.to_datetime(xray["time"].values, utc=True)
        chi = solar_zenith_angle(lat, lon, times, method=self.sza_method)
        phi = xray["xrsb"].values

        L = self._absorption(f, chi, phi)

        return pd.DataFrame(
            {"sza_deg": chi, "xrsb": phi, "absorption_db": L},
            index=pd.Index(times, name="time"),
        )

    def predict_point(self, lat: float, lon: float, time, xray_wm2: float,
                      *, freq_mhz: float | None = None) -> float:
        """Single-sample convenience wrapper around :func:`xrap.absorption`."""
        f = freq_mhz if freq_mhz is not None else self.freq_mhz
        chi = solar_zenith_angle(lat, lon, time, method=self.sza_method)
        L = self._absorption(f, chi, xray_wm2)
        return float(np.ravel(L)[0])
