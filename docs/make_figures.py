"""Regenerate the figures embedded in docs/model.md.

Needs network (SunPy + SWPC) and matplotlib::

    python docs/make_figures.py

Outputs to docs/images/. The "last 24 h" panels change with each run; the
2017-09-06 panels are fixed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xrap import (
    AbsorptionModel,
    absorption,
    fetch_goes_xrs,
    plot_absorption,
    plot_xrs,
)

IMG = Path(__file__).parent / "images"
IMG.mkdir(exist_ok=True)
COLORS = {"xrap": "#2ca02c", "drap2": "#1f77b4"}


def _both(ds, lat, lon, freq=10.0):
    return {
        m: AbsorptionModel(freq_mhz=freq, model=m, sza_method="noaa").predict(
            lat=lat, lon=lon, start=ds.time.values[0], end=ds.time.values[-1], xray=ds
        )
        for m in ("xrap", "drap2")
    }


def _overlay(dfs, title, name):
    ax = None
    for model, df in dfs.items():
        ax = plot_absorption(df, ax=ax, label=model.upper(), color=COLORS[model], title=title)
    ax.figure.savefig(IMG / name, dpi=120)
    plt.close(ax.figure)


def main() -> None:
    # GOES flux + absorption overlay for the 2017-09-06 flares
    ds = fetch_goes_xrs("2017-09-06T09:00", "2017-09-06T15:00", source="sunpy", satellite=16)
    ax = plot_xrs(ds, title="GOES-16 XRS  —  2017-09-06 (X2.2 at 09:10, X9.3 at 12:02 UT)")
    ax.figure.savefig(IMG / "goes_xrs_2017-09-06.png", dpi=120)
    plt.close(ax.figure)
    _overlay(
        _both(ds, 20.0, 30.0),
        "10 MHz one-way absorption  —  2017-09-06 flare, 20°N 30°E",
        "absorption_2017-09-06_10MHz_models.png",
    )

    # last 24 h
    ld = fetch_goes_xrs(source="noaa_json", feed="1-day")
    _overlay(
        _both(ld, 0.0, 0.0),
        "10 MHz one-way absorption  —  last 24 h, 0°N 0°E",
        "absorption_last24h_10MHz_models.png",
    )

    # absorption vs flux, both models, 10 & 30 MHz
    flux = np.logspace(-7, -3, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f, ls in ((30.0, "-"), (10.0, "--")):
        ax.loglog(flux, absorption(f, 0.0, flux, model="xrap", grazing=False),
                  ls, color=COLORS["xrap"], label=f"XRAP  {f:g} MHz")
        ax.loglog(flux, absorption(f, 0.0, flux, model="drap2"),
                  ls, color=COLORS["drap2"], label=f"DRAP2 {f:g} MHz")
    for x, cls in ((1e-6, "C"), (1e-5, "M"), (1e-4, "X")):
        ax.axvline(x, color="0.85", lw=0.7, zorder=0)
        ax.text(x, 1.3e-3, cls, ha="center", fontsize=9, color="0.4")
    ax.set_xlabel(r"GOES 0.1-0.8 nm flux  [W m$^{-2}$]")
    ax.set_ylabel("one-way absorption, overhead Sun  [dB]")
    ax.set_title("XRAP vs DRAP2:  absorption vs flare magnitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(IMG / "models_vs_flux.png", dpi=120)
    plt.close(fig)

    print(f"wrote figures to {IMG}")


if __name__ == "__main__":
    main()
