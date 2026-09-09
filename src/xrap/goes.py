"""GOES/XRS soft X-ray flux retrieval.

The model needs a time series of solar soft X-ray irradiance (W m-2): the GOES
XRS "long" band (0.1-0.8 nm, ``xrsb``) and the "short" band (0.05-0.4 nm,
``xrsa``).

Retrieval routes (``source=``)
------------------------------
``"sunpy"`` (default)
    SunPy ``Fido`` search of the NOAA XRS collection. Science-quality,
    inter-calibrated data for any event from ~2010 onward (GOES 13-18).
    ``resolution="1min"`` -> ``avg1m`` netCDF; ``"1s"`` -> ``flx1s``.

``"noaa_json"``
    NOAA SWPC operational JSON feeds -- near-real-time, **last ~7 days only**:

    * ``xrays-6-hour`` / ``xrays-1-day`` / ``xrays-3-day`` / ``xrays-7-day`` --
      1-min flux for both energy bands. We read the ``observed_flux`` field
      (raw sensor irradiance) for each band. The narrowest feed covering the
      request is used.
    * :func:`fetch_goes_flares` reads ``xray-flares-7-day`` /
      ``xray-flares-latest`` for flare event summaries (begin/max/end times and
      classes).

``"netcdf_url"`` / ``"local"``
    Direct download or a local file for **prior events** (pre-2010, or
    reprocessed archives). Not implemented yet -- the user supplies exact
    URLs/paths per event.

All routes return the same :class:`xarray.Dataset` so downstream code is
source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr

__all__ = ["fetch_goes_xrs", "fetch_goes_flares", "XRSResult"]

Source = Literal["sunpy", "noaa_json", "netcdf_url", "local"]

#: SWPC operational JSON feeds.
_SWPC_BASE = "https://services.swpc.noaa.gov/json/goes"
_XRAYS_FEEDS = (  # (feed key, filename, max age it covers) -- widening order
    ("6-hour", "xrays-6-hour.json", pd.Timedelta(hours=6)),
    ("1-day", "xrays-1-day.json", pd.Timedelta(days=1)),
    ("3-day", "xrays-3-day.json", pd.Timedelta(days=3)),
    ("7-day", "xrays-7-day.json", pd.Timedelta(days=7)),
)
_XRAYS_MAX_AGE = pd.Timedelta(days=7)

#: XRS energy-band labels used in the SWPC JSON.
_BAND_XRSA = "0.05-0.4nm"
_BAND_XRSB = "0.1-0.8nm"

_HTTP_TIMEOUT = 30  # seconds


@dataclass
class XRSResult:
    """A retrieved XRS time series.

    Attributes
    ----------
    data : xarray.Dataset
        ``xrsa`` (0.05-0.4 nm) and ``xrsb`` (0.1-0.8 nm) in W m-2, on a UTC
        ``time`` coordinate. May carry ``xrsa_quality`` / ``xrsb_quality``.
    satellite : str
        e.g. ``"GOES-16"``.
    source : str
        Retrieval route.
    quality : str
        ``"science"`` (re-processed) or ``"nrt"`` (operational).
    """

    data: xr.Dataset
    satellite: str
    source: str
    quality: str


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fetch_goes_xrs(
    start=None,
    end=None,
    *,
    source: Source = "sunpy",
    satellite: str | int | None = None,
    resolution: str = "1min",
    feed: str | None = None,
    cache: bool = True,
    url: str | None = None,
    path: str | None = None,
) -> xr.Dataset:
    """Fetch a GOES/XRS soft X-ray flux time series.

    Parameters
    ----------
    start, end : str or datetime, optional
        UTC bounds of the interval. Required for every source except
        ``noaa_json`` with an explicit ``feed=`` (then they default to the
        whole feed).
    source : {"sunpy", "noaa_json", "netcdf_url", "local"}
        Retrieval route (see module docstring).
    satellite : str or int, optional
        Preferred GOES satellite (``16`` / ``"GOES-16"``). ``None`` lets the
        route pick (highest number available for ``sunpy``; the SWPC primary
        for ``noaa_json``). ``"secondary"`` selects the SWPC secondary feed.
    resolution : {"1min", "1s"}
        Cadence for ``source="sunpy"`` (``avg1m`` / ``flx1s``). Ignored by
        ``noaa_json`` (always 1-min).
    feed : {"6-hour", "1-day", "3-day", "7-day"}, optional
        ``noaa_json`` only -- force a specific SWPC feed instead of auto-picking
        the narrowest one that covers ``[start, end]``.
    cache : bool
        Cache SunPy downloads under :func:`_cache_dir`.
    url, path : str, optional
        Required for ``source="netcdf_url"`` and ``source="local"``.

    Returns
    -------
    xarray.Dataset
        ``xrsa`` / ``xrsb`` in W m-2 on a UTC ``time`` index. Gaps are kept as
        NaN, not interpolated. ``.attrs`` records ``satellite``, ``source``,
        ``quality``, band definitions and units.
    """
    open_ended = source == "noaa_json" and feed is not None
    start = pd.to_datetime(start, utc=True) if start is not None else (
        pd.Timestamp("1970-01-01", tz="UTC") if open_ended else None
    )
    end = pd.to_datetime(end, utc=True) if end is not None else (
        pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1) if open_ended else None
    )
    if start is None or end is None:
        raise ValueError("start and end are required (except noaa_json with feed=)")
    if start >= end:
        raise ValueError("start must be before end")

    if source == "sunpy":
        res = _fetch_sunpy(start, end, satellite, resolution, cache)
    elif source == "noaa_json":
        res = _fetch_noaa_json(start, end, satellite, feed)
    elif source == "netcdf_url":
        if not url:
            raise ValueError("source='netcdf_url' requires url=")
        res = _fetch_netcdf_url(start, end, url, cache)
    elif source == "local":
        if not path:
            raise ValueError("source='local' requires path=")
        res = _load_local(start, end, path)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown source {source!r}")

    res.data.attrs.update(
        satellite=res.satellite, source=res.source, quality=res.quality
    )
    return res.data


def fetch_goes_flares(
    *,
    latest: bool = False,
    spacecraft: str = "primary",
    start=None,
    end=None,
) -> pd.DataFrame:
    """Flare event summaries from the SWPC ``xray-flares`` JSON feeds.

    Parameters
    ----------
    latest : bool
        ``True`` -> ``xray-flares-latest.json`` (the single most recent event).
        ``False`` (default) -> ``xray-flares-7-day.json`` (all events, ~7 days).
    spacecraft : {"primary", "secondary"}
        Which SWPC feed.
    start, end : str or datetime, optional
        If given, keep only flares whose ``max_time`` falls in ``[start, end]``.

    Returns
    -------
    pandas.DataFrame
        One row per flare. ``begin_time`` / ``max_time`` / ``end_time`` parsed
        to UTC ``datetime64``; ``max_class`` (e.g. ``"X2.1"``), ``max_xrlong``
        (peak 0.1-0.8 nm flux, W m-2), ``satellite``, plus the other feed
        fields. Indexed by ``max_time``.
    """
    name = "xray-flares-latest.json" if latest else "xray-flares-7-day.json"
    records = _get_json(f"{_SWPC_BASE}/{spacecraft}/{name}")

    df = pd.DataFrame(records)
    for col in ("begin_time", "max_time", "end_time", "time_tag", "max_ratio_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "max_time" in df:
        df = df.sort_values("max_time").set_index("max_time", drop=False)
    if start is not None and end is not None and "max_time" in df:
        s, e = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        df = df[(df["max_time"] >= s) & (df["max_time"] <= e)]
    return df


# --------------------------------------------------------------------------- #
# Route: SunPy
# --------------------------------------------------------------------------- #
def _fetch_sunpy(start, end, satellite, resolution, cache) -> XRSResult:
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from sunpy.timeseries import TimeSeries

    res = {"1min": "avg1m", "avg1m": "avg1m", "1m": "avg1m",
           "1s": "flx1s", "flx1s": "flx1s"}.get(str(resolution))
    if res is None:
        raise ValueError(f"resolution must be '1min' or '1s', got {resolution!r}")

    sat_no = _sat_number(satellite)
    query = [
        a.Time(_naive_utc(start), _naive_utc(end)),
        a.Instrument.xrs,
        a.Resolution(res),
    ]
    if sat_no is not None:
        query.append(a.goes.SatelliteNumber(sat_no))

    results = Fido.search(*query)
    table = results[0] if len(results) else None
    if table is None or len(table) == 0:
        raise ValueError(
            f"no GOES/XRS {res} data for {start:%Y-%m-%d %H:%M}..{end:%Y-%m-%d %H:%M}"
            + ("" if sat_no is None else f" (satellite {sat_no})")
        )

    available = sorted({int(n) for n in table["SatelliteNumber"]})
    chosen = sat_no if sat_no in available else available[-1]  # newest by default
    rows = [i for i, n in enumerate(table["SatelliteNumber"]) if int(n) == chosen]

    fetch_path = str(_cache_dir() / "{file}") if cache else None
    files = Fido.fetch(table[rows], path=fetch_path)
    if getattr(files, "errors", None):
        raise RuntimeError(f"SunPy download failed: {files.errors}")

    ts = TimeSeries(sorted(map(str, files)), concatenate=True)
    df = ts.to_dataframe().sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df = df.loc[(df.index >= start) & (df.index <= end)]
    if "xrsa" not in df or "xrsb" not in df:
        raise RuntimeError(f"unexpected XRS columns: {list(df.columns)}")

    data = _to_dataset(
        df.index,
        df["xrsa"].to_numpy(float),
        df["xrsb"].to_numpy(float),
        xrsa_quality=df["xrsa_quality"].to_numpy() if "xrsa_quality" in df else None,
        xrsb_quality=df["xrsb_quality"].to_numpy() if "xrsb_quality" in df else None,
    )
    return XRSResult(data, satellite=f"GOES-{chosen}", source="sunpy", quality="science")


# --------------------------------------------------------------------------- #
# Route: NOAA SWPC operational JSON
# --------------------------------------------------------------------------- #
def _fetch_noaa_json(start, end, satellite, feed=None) -> XRSResult:
    if feed is not None:
        fname = _resolve_feed(feed)
    else:
        age = pd.Timestamp.now(tz="UTC") - start
        if age > _XRAYS_MAX_AGE:
            raise ValueError(
                "source='noaa_json' only covers the last ~7 days; "
                "use source='sunpy' for older events"
            )
        fname = next(f for _, f, span in _XRAYS_FEEDS if age <= span)
    spacecraft = "secondary" if str(satellite).lower() == "secondary" else "primary"

    records = _get_json(f"{_SWPC_BASE}/{spacecraft}/{fname}")
    df = pd.DataFrame(records)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

    # observed_flux per band, pivoted onto a common time axis
    wide = (
        df.pivot_table(index="time_tag", columns="energy", values="observed_flux")
        .sort_index()
    )
    time = wide.index
    xrsa = wide[_BAND_XRSA].to_numpy(float) if _BAND_XRSA in wide else np.full(len(time), np.nan)
    xrsb = wide[_BAND_XRSB].to_numpy(float) if _BAND_XRSB in wide else np.full(len(time), np.nan)

    mask = (time >= start) & (time <= end)
    sat = int(pd.Series(df["satellite"]).mode().iloc[0]) if "satellite" in df else 0
    data = _to_dataset(time[mask], xrsa[mask], xrsb[mask])
    return XRSResult(
        data, satellite=f"GOES-{sat}" if sat else "GOES", source="noaa_json", quality="nrt"
    )


# --------------------------------------------------------------------------- #
# Route: archived / local (prior events) -- fill in
# --------------------------------------------------------------------------- #
def _fetch_netcdf_url(start, end, url, cache) -> XRSResult:
    """Download a GOES science-data netCDF (e.g. NOAA NCEI) and slice it.

    Archive root:
      https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/
    """
    raise NotImplementedError


def _load_local(start, end, path) -> XRSResult:
    """Load a user-provided local file (netCDF / CSV) for a prior event."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_dataset(time, xrsa, xrsb, *, xrsa_quality=None, xrsb_quality=None) -> xr.Dataset:
    """Assemble the canonical XRS :class:`xarray.Dataset`."""
    # plain, name-less UTC datetime64 axis (avoids a stray coord dimension when
    # the source index is named e.g. "time_tag")
    time = pd.to_datetime(time, utc=True).tz_convert("UTC").tz_localize(None)
    time = np.asarray(pd.DatetimeIndex(time).values, dtype="datetime64[ns]")
    variables = {
        "xrsa": ("time", np.asarray(xrsa, dtype=float)),
        "xrsb": ("time", np.asarray(xrsb, dtype=float)),
    }
    if xrsa_quality is not None:
        variables["xrsa_quality"] = ("time", np.asarray(xrsa_quality))
    if xrsb_quality is not None:
        variables["xrsb_quality"] = ("time", np.asarray(xrsb_quality))
    return xr.Dataset(
        variables,
        coords={"time": time},
        attrs={
            "xrsa_band_nm": "0.05-0.4",
            "xrsb_band_nm": "0.1-0.8",
            "units": "W m-2",
        },
    )


def _get_json(url: str):
    """GET a JSON document, raising for HTTP errors."""
    resp = requests.get(url, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _resolve_feed(feed: str) -> str:
    """``"6-hour"`` / ``"1d"`` / ``"xrays-3-day.json"`` -> canonical filename."""
    key = (
        str(feed).lower().strip()
        .removesuffix(".json").removeprefix("xrays-")
        .replace("hours", "h").replace("hour", "h")
        .replace("days", "d").replace("day", "d")
        .replace("-", "")
    )
    table = {"6h": "xrays-6-hour.json", "1d": "xrays-1-day.json",
             "3d": "xrays-3-day.json", "7d": "xrays-7-day.json"}
    if key not in table:
        raise ValueError(
            f"feed must be one of 6-hour, 1-day, 3-day, 7-day; got {feed!r}"
        )
    return table[key]


def _sat_number(satellite) -> int | None:
    """Parse ``16`` / ``"16"`` / ``"GOES-16"`` -> ``16``; ``None``/``"primary"``
    /``"secondary"`` -> ``None``."""
    if satellite is None:
        return None
    s = str(satellite).lower().replace("goes", "").replace("-", "").strip()
    if s in ("", "primary", "secondary"):
        return None
    return int(s)


def _naive_utc(ts: pd.Timestamp):
    """tz-aware UTC Timestamp -> naive datetime (what ``sunpy.net.attrs.Time`` wants)."""
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).to_pydatetime()


def _cache_dir():
    """On-disk cache directory for downloads (``$XRAP_CACHE`` or ``~/.xrap/cache``)."""
    import os
    from pathlib import Path

    root = Path(os.environ.get("XRAP_CACHE", Path.home() / ".xrap" / "cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root
