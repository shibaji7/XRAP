"""Quick-look plots.

Optional -- needs matplotlib (``pip install "xrap[plot]"``). Imported lazily so
the rest of the package has no hard matplotlib dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["plot_xrs", "plot_absorption"]


def _mpl_times(values):
    """Coerce a time axis to tz-naive ``datetime64[ns]`` matplotlib can plot.

    Handles tz-aware indexes and object arrays of Timestamps, which otherwise
    trip an ``OverflowError`` in older matplotlib date handling.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.tz_convert("UTC").tz_localize(None).to_numpy()

#: GOES flare classes and their 0.1-0.8 nm flux thresholds (W m-2).
_FLARE_CLASSES = [("A", 1e-8), ("B", 1e-7), ("C", 1e-6), ("M", 1e-5), ("X", 1e-4)]

_BAND_LABEL = {"xrsa": "0.05-0.4 nm (short)", "xrsb": "0.1-0.8 nm (long)"}
_BAND_COLOR = {"xrsa": "#1f77b4", "xrsb": "#d62728"}


def plot_xrs(
    ds,
    *,
    ax=None,
    bands=("xrsa", "xrsb"),
    flare_classes: bool = True,
    ylim=(1e-9, 1e-2),
    title: str | None = None,
):
    """Plot GOES/XRS flux (log y) versus time.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`xrap.fetch_goes_xrs` (``xrsa`` / ``xrsb`` vs ``time``).
    ax : matplotlib.axes.Axes, optional
        Draw on this axis; a new figure is made otherwise.
    bands : sequence of str
        Which variables to draw.
    flare_classes : bool
        Draw the A/B/C/M/X reference lines and right-margin labels.
    ylim : tuple
        Flux axis limits (W m-2).
    title : str, optional
        Overrides the default ``"<satellite> XRS (<source>)"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    t = _mpl_times(ds["time"].values)
    for b in bands:
        if b in ds:
            y = np.asarray(ds[b].values, dtype=float)
            y = np.where(y > 0.0, y, np.nan)  # break the line across gaps / zeros
            ax.plot(t, y, lw=1.0, label=_BAND_LABEL.get(b, b),
                    color=_BAND_COLOR.get(b))

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel(r"X-ray flux  [W m$^{-2}$]")
    ax.set_xlabel("Time  [UTC]")

    if flare_classes:
        for name, level in _FLARE_CLASSES:
            if ylim[0] <= level <= ylim[1]:
                ax.axhline(level, color="0.85", lw=0.6, zorder=0)
                ax.text(1.01, level, name, transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=8, color="0.5")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    sat = ds.attrs.get("satellite", "GOES")
    src = ds.attrs.get("source", "")
    ax.set_title(title or f"{sat} XRS" + (f"  ({src})" if src else ""))
    ax.figure.tight_layout()
    return ax


def plot_absorption(df, *, ax=None, column="absorption_db", label=None, color=None,
                    freq_label=None, title=None):
    """Plot absorption (dB) versus time.

    Call it more than once with the same ``ax`` (and different ``label=``) to
    overlay models -- e.g. ``"xrap"`` and ``"drap2"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :meth:`xrap.AbsorptionModel.predict` -- a UTC ``time`` index
        and an ``absorption_db`` column.
    ax : matplotlib.axes.Axes, optional
        Draw on this axis; a new figure is made otherwise.
    column : str
        Column to plot.
    label : str, optional
        Trace label; when given on any call, a legend is shown.
    color : str, optional
        Line colour; defaults to the matplotlib cycle.
    freq_label : str, optional
        Appended to the default title, e.g. ``"10 MHz"``.
    title : str, optional
        Overrides the default title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.plot(_mpl_times(df.index.values), np.asarray(df[column].values, dtype=float),
            lw=1.2, label=label, color=color)
    # y from 0 up to the max of *all* traces (so overlaid models both fit)
    ymax = max(
        (float(np.nanmax(ln.get_ydata())) for ln in ax.get_lines() if ln.get_ydata().size),
        default=1.0,
    )
    ax.set_ylim(0.0, ymax * 1.08 if np.isfinite(ymax) and ymax > 0 else 1.0)
    ax.set_ylabel("D-region absorption  [dB]")
    ax.set_xlabel("Time  [UTC]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(title or "HF absorption" + (f"  ({freq_label})" if freq_label else ""))

    handles = [ln for ln in ax.get_lines() if (ln.get_label() or "_").startswith("_") is False]
    if handles:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.figure.tight_layout()
    return ax
