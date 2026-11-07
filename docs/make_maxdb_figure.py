"""Plot oblique absorption with ``max_db`` peak scaling applied.

Needs network (SunPy) + matplotlib::

    python docs/make_maxdb_figure.py              # peak scaled to 100 dB (default)
    python docs/make_maxdb_figure.py --max-db 60  # e.g. 60 dB

Shows the 2017-09-06 X9.3 flare on a 1-hop, ~2200 km circuit at 10 MHz and at
3.75 MHz (80 m). The solid curves are ``oblique_absorption(..., max_db=...)``:
each whole series is multiplied by one factor so its **peak equals** ``max_db``
(the shape is kept; no flat top). The faint dotted curves are the raw values
(mostly off-scale; peak and scale factor are in the legend). Because each series
is scaled by its own factor, the two curves are *not* comparable in magnitude.
The right-hand axis is the same curve as a fraction of ``max_db``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrap import fetch_goes_xrs, oblique_absorption

IMG = Path(__file__).parent / "images"
IMG.mkdir(exist_ok=True)

TX, RX = (35.0, 25.0), (48.0, 5.0)  # 1 hop, ~2200 km
HEIGHT_KM = 300.0
CURVES = ((10.0, "#d62728", "10 MHz"), (3.75, "#1f77b4", "3.75 MHz (80 m)"))


def main(max_db: float) -> None:
    # GOES-15 = the satellite behind the official X9.3 class (flux on the class scale)
    ds = fetch_goes_xrs("2017-09-06T11:00", "2017-09-06T15:00", source="sunpy", satellite=15)
    t = pd.to_datetime(ds.time.values, utc=True)
    flux = ds.xrsb.values

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for freq, color, name in CURVES:
        raw = oblique_absorption(TX, RX, t, freq, flux, n_hops=1, height_km=HEIGHT_KM)
        scaled = oblique_absorption(TX, RX, t, freq, flux, n_hops=1, height_km=HEIGHT_KM,
                                    max_db=max_db)
        factor = np.nanmax(scaled) / np.nanmax(raw)
        ax.plot(t, raw, color=color, lw=0.8, ls=":", alpha=0.5)
        ax.plot(t, scaled, color=color, lw=1.6,
                label=f"{name}  (raw peak {np.nanmax(raw):,.0f} dB  ->  x{factor:.3f})")

    ax.axhline(max_db, color="0.25", lw=1.0, ls="--")
    ax.text(t[0], max_db, f"  max_db = {max_db:g} dB  =  the peak of each scaled series",
            va="bottom", ha="left", fontsize=8, color="0.25")
    ax.text(0.015, 0.40,
            "The two solid curves coincide: the 10 MHz and 3.75 MHz series differ only\n"
            "by a constant factor, so scaling each to its own peak makes them identical\n"
            "-- peak scaling discards the frequency (and any cross-series) information.",
            transform=ax.transAxes, fontsize=8, va="center", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))
    ax.set_ylim(0, max_db * 1.12)
    ax.set_ylabel("oblique absorption  [dB]   (series scaled so its peak = max_db)")
    ax.set_xlabel("Time  [UTC]")
    ax.set_title(f"Oblique absorption, peak scaled to max_db = {max_db:g} dB  -  "
                 "2017-09-06 X9.3 (GOES-15), 1-hop ~2200 km")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.005, 0.87))

    sec = ax.secondary_yaxis("right", functions=(lambda a: a / max_db, lambda f: f * max_db))
    sec.set_ylabel("fraction of max_db")

    fig.tight_layout()
    out = IMG / "oblique_max_db.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-db", type=float, default=100.0, help="peak to scale each series to, dB (default 100)")
    main(ap.parse_args().max_db)
