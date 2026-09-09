"""Command-line interface: ``xrap ...``.

Thin wrapper over :func:`xrap.goes.fetch_goes_xrs` and
:class:`xrap.model.AbsorptionModel`.

Examples
--------
    # DRAP2 absorption time series at a point
    xrap predict --lat 40 --lon -105 --freq 10 \
        --start 2017-09-06T11:00 --end 2017-09-06T13:00 -o absorption.csv

    # ... with a quick-look plot
    xrap predict --lat 20 --lon 30 --freq 10 \
        --start 2017-09-06T09:00 --end 2017-09-06T15:00 --plot absorption.png

    # just the GOES/XRS flux
    xrap fetch-goes --start 2017-09-06T11:00 --end 2017-09-06T13:00 -o xrs.nc
    xrap fetch-goes --source noaa_json --feed 1-day --plot xrs.png
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xrap", description="X-Ray Absorption Prediction (DRAP2 solar-flare)")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict", help="DRAP2 absorption time series at a location")
    pp.add_argument("--lat", type=float, required=True, help="latitude (deg)")
    pp.add_argument("--lon", type=float, required=True, help="longitude (deg, E-positive)")
    pp.add_argument("--freq", type=float, default=10.0, help="HF frequency (MHz)")
    pp.add_argument("--start", required=True, help="UTC start")
    pp.add_argument("--end", required=True, help="UTC end")
    pp.add_argument("--model", default="xrap", choices=("xrap", "drap2"))
    pp.add_argument("--goes-source", default="sunpy", choices=("sunpy", "noaa_json"))
    pp.add_argument("--satellite", default=None, help="e.g. 16 or GOES-16")
    pp.add_argument("--feed", default=None, help="noaa_json feed: 6-hour|1-day|3-day|7-day")
    pp.add_argument("--sza-method", default="noaa", choices=("noaa", "astropy"))
    pp.add_argument("--path", default="oneway", choices=("oneway", "total"))
    pp.add_argument(
        "--grazing", action=argparse.BooleanOptionalAction, default=None,
        help="Chapman grazing correction (default: on for xrap, off for drap2)",
    )
    pp.add_argument("-o", "--output", help="write CSV here (default: stdout)")
    pp.add_argument("--plot", metavar="PNG", help="also save an absorption-vs-time plot")

    pg = sub.add_parser("fetch-goes", help="download a GOES/XRS flux series")
    pg.add_argument("--start", default=None, help="UTC start (optional with --feed)")
    pg.add_argument("--end", default=None, help="UTC end (optional with --feed)")
    pg.add_argument("--source", default="sunpy", choices=("sunpy", "noaa_json"))
    pg.add_argument("--satellite", default=None, help="e.g. 16 or GOES-16")
    pg.add_argument("--resolution", default="1min", choices=("1min", "1s"))
    pg.add_argument("--feed", default=None, help="noaa_json feed: 6-hour|1-day|3-day|7-day")
    pg.add_argument("-o", "--output", help="output path (.nc -> netCDF, else CSV)")
    pg.add_argument("--plot", metavar="PNG", help="also save a flux-vs-time plot")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "fetch-goes":
        return _cmd_fetch_goes(args)
    return 1  # pragma: no cover - argparse enforces a subcommand


def _cmd_predict(args) -> int:
    from .model import AbsorptionModel

    model = AbsorptionModel(
        freq_mhz=args.freq,
        model=args.model,
        sza_method=args.sza_method,
        goes_source=args.goes_source,
        path=args.path,
        grazing=args.grazing,
    )
    goes_kwargs = {}
    if args.satellite is not None:
        goes_kwargs["satellite"] = args.satellite
    if args.feed is not None:
        goes_kwargs["feed"] = args.feed

    df = model.predict(args.lat, args.lon, args.start, args.end, **goes_kwargs)

    if args.output:
        df.to_csv(args.output)
        print(f"wrote {len(df)} rows -> {args.output}", file=sys.stderr)
    else:
        df.to_csv(sys.stdout)

    if args.plot:
        from .plot import plot_absorption

        ax = plot_absorption(df, freq_label=f"{args.freq:g} MHz")
        ax.figure.savefig(args.plot, dpi=120)
        print(f"wrote plot -> {args.plot}", file=sys.stderr)
    return 0


def _cmd_fetch_goes(args) -> int:
    from .goes import fetch_goes_xrs

    ds = fetch_goes_xrs(
        args.start,
        args.end,
        source=args.source,
        satellite=args.satellite,
        resolution=args.resolution,
        feed=args.feed,
    )

    out = args.output
    if out:
        if out.endswith(".nc"):
            ds.to_netcdf(out)
        else:
            ds.to_dataframe().to_csv(out)
        print(f"wrote {ds.sizes['time']} samples -> {out}", file=sys.stderr)
    else:
        ds.to_dataframe().to_csv(sys.stdout)

    if args.plot:
        from .plot import plot_xrs

        ax = plot_xrs(ds)
        ax.figure.savefig(args.plot, dpi=120)
        print(f"wrote plot -> {args.plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
