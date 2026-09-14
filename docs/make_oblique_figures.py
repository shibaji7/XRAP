"""Regenerate the figure(s) embedded in docs/oblique.md.

Needs network (SunPy) + matplotlib::

    python docs/make_oblique_figures.py

Isolates the secant-law obliquity factor on a real event: computes the
oblique total for a 1-hop circuit against the *fair* baseline (the sum of the
same two D-region crossings' vertical absorption, with the obliquity factor
stripped out), and checks the ratio against sec(incidence)**2.5 analytically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrap import (
    absorption,
    fetch_goes_xrs,
    hop_crossings,
    oblique_absorption,
    obliquity_factor,
    solar_zenith_angle,
)

IMG = Path(__file__).parent / "images"
IMG.mkdir(exist_ok=True)

# A 1-hop, ~2200 km, mid-latitude circuit (e.g. southern Greece -> N.E. France)
TX, RX = (35.0, 25.0), (48.0, 5.0)
FREQ_MHZ = 10.0
HEIGHT_KM = 300.0


def main() -> None:
    ds = fetch_goes_xrs("2017-09-06T09:00", "2017-09-06T15:00", source="sunpy", satellite=16)
    t = pd.to_datetime(ds.time.values, utc=True)
    flux = ds.xrsb.values

    crossings = hop_crossings(TX, RX, n_hops=1, height_km=HEIGHT_KM)
    incidence = crossings[0].incidence_deg  # both legs share one incidence (see docs/oblique.md)
    obl = obliquity_factor(incidence, n=1.5)

    # Fair baseline: sum of the two real crossings' vertical (obliquity-free)
    # absorption -- NOT a single midpoint, which mixes in an unrelated SZA.
    vertical_sum = np.zeros_like(flux, dtype=float)
    for c in crossings:
        chi = solar_zenith_angle(c.lat, c.lon, t, method="noaa")
        vertical_sum += absorption(FREQ_MHZ, chi, flux, model="xrap", path="oneway", grazing=False)

    oblique = oblique_absorption(TX, RX, t, FREQ_MHZ, flux, n_hops=1, height_km=HEIGHT_KM)

    peak = int(np.nanargmax(flux))
    measured_ratio = oblique[peak] / vertical_sum[peak]
    print(f"incidence = {incidence:.2f} deg")
    print(f"sec(incidence)^2.5 (predicted) = {obl:.3f}")
    print(f"oblique/vertical_sum at flux peak (measured) = {measured_ratio:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, oblique, color="#d62728", lw=1.3,
            label=rf"oblique total  (2 crossings $\times\ \sec^{{2.5}}$"
                  rf"({incidence:.0f}$^\circ$) = {obl:.1f}$\times$)")
    ax.plot(t, vertical_sum, color="#2ca02c", lw=1.3, ls="--",
            label="sum of the 2 crossings' vertical absorption  (obliquity factor = 1)")
    ax.set_ylabel("absorption  [dB]")
    ax.set_xlabel("Time  [UTC]")
    ax.set_title("Obliquity factor isolated -- 2017-09-06 X9.3, 1-hop circuit, 10 MHz")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "oblique_vs_vertical_2017.png", dpi=120)
    plt.close(fig)
    print(f"wrote {IMG / 'oblique_vs_vertical_2017.png'}")


if __name__ == "__main__":
    main()
