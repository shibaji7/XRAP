# XRAP — X-Ray Absorption Prediction

[![CI](https://github.com/shibaji7/XRAP/actions/workflows/ci.yml/badge.svg)](https://github.com/shibaji7/XRAP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shibaji7/XRAP/branch/main/graph/badge.svg)](https://codecov.io/gh/shibaji7/XRAP)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Estimate solar-flare-driven **HF radio-wave absorption** for a specific HF
frequency, parameterized by:

- **solar zenith angle (SZA)** at the absorbing point,
- **HF frequency**,
- **solar soft X-ray flux** (GOES/XRS, 0.1–0.8 nm long band).

Two absorption models are available, selected with `model=`:

**`"xrap"` (default)** — Fiori, Chakraborty & Nikitina (2022), a data-optimized
model fit to the NRCan 30 MHz riometer network
([doi:10.1016/j.jastp.2022.105843](https://doi.org/10.1016/j.jastp.2022.105843)):

```
A(f0) = 12080 * flux[W/m^2] * cos(chi)     # dB, one-way, f0 = 30 MHz, A >= 0
A(f)  = (f0 / f)^1.5 * A(f0)
```

**`"drap2"`** — NOAA Global D-Region Absorption Prediction v2, X-ray term
([documentation](https://www.spaceweather.gov/content/global-d-region-absorption-prediction-documentation)):

```
HAF0     = 10 * log10(flux[W/m^2]) + 65    # MHz, sub-solar
HAF(chi) = HAF0 * (cos chi)^0.75           # 0 for chi >= 90 deg
A(f,chi) = 0.5 * (HAF(chi) / f)^1.5        # dB, one-way
```

`flux` is the GOES 0.1–0.8 nm long band. Near the terminator the `cos(chi)`
factor is either hard-clipped at 90° (`grazing=False`) or replaced by the
Chapman grazing function `1/Ch(chi)` (`grazing=True`, the default for `"xrap"`)
— a small positive value that decays smoothly to zero at the true terminator
(~99.6° at 90 km).

**[`docs/model.md`](docs/model.md)** has the full derivations, the grazing
treatment, and a worked **XRAP-vs-DRAP2 comparison** (with figures) showing why
DRAP2 underestimates strong flares.

> Status: `sza`, both `absorption` models, the `sunpy` / `noaa_json` GOES routes,
> the `xrap` CLI, and the plot helpers are implemented and tested. Still stubbed:
> the `netcdf_url` / `local` GOES routes (pre-2010 / archived events) and
> `polar_cap_absorption` (DRAP2 SEP term) — both raise `NotImplementedError`.

## Install (from GitHub, into a local environment)

```bash
git clone https://github.com/shibaji7/XRAP.git
cd XRAP
pip install .
```

Or without cloning:

```bash
pip install "git+https://github.com/shibaji7/XRAP.git"
```

`pip` reads [`pyproject.toml`](pyproject.toml) and installs all runtime
dependencies (numpy, scipy, pandas, xarray, astropy, sunpy, requests …)
automatically. Optional extras:

```bash
pip install "xrap[plot]"      # matplotlib helpers
pip install "xrap[netcdf]"    # netCDF4/h5netcdf for archived GOES files
pip install "xrap[test]"      # test + coverage tooling
```

## Local conda development environment

```bash
make env          # conda env create -f environment.yml
conda activate xrap
make install      # pip install -e ".[dev]"
```

Everything stays inside the `xrap` conda env; nothing is installed
system-wide. Run `make help` for the full task list (`test`, `coverage`,
`lint`, `format`, `build`, `clean`).

## Quick start

```python
import xrap

# 1. SZA from lat/lon/time
chi = xrap.solar_zenith_angle(lat=40.0, lon=-105.0, time="2017-09-06T12:02Z")

# 2. GOES X-ray flux (SunPy; any flare from 2010 onward)
xrs = xrap.fetch_goes_xrs("2017-09-06T11:00", "2017-09-06T13:00")

# 3. Absorption in dB
dB = xrap.absorption(freq_mhz=10.0, sza_deg=chi, xray_wm2=xrs["xrsb"])  # model="xrap"
dB_drap2 = xrap.absorption(10.0, chi, xrs["xrsb"], model="drap2")
#   path="oneway" (default) | "total"
#   grazing=True/False  -> override the per-model default
#   alt_km=90, H_km=7   -> grazing geometry (overridable)

# XRAP one-way value at the 30 MHz reference frequency
a30 = xrap.reference_absorption(sza_deg=chi, xray_wm2=xrs["xrsb"])

# rescale an absorption value to another frequency:  A(f) = (f0/f)^n * A(f0)
dB_20 = xrap.scale_frequency(dB, f0_mhz=10.0, f_mhz=20.0)        # n=1.5
dB_20 = xrap.scale_frequency(dB, f0_mhz=10.0, f_mhz=20.0, n=2.0) # custom exponent

# …or end to end
model = xrap.AbsorptionModel(freq_mhz=10.0, model="xrap")
df = model.predict(lat=40.0, lon=-105.0,
                   start="2017-09-06T11:00", end="2017-09-06T13:00")
```

### GOES data sources

| `source`     | Coverage                        | Notes                                    |
|--------------|---------------------------------|------------------------------------------|
| `sunpy`      | ~2010 → present                 | science-quality, via `Fido`; `resolution="1min"`/`"1s"` (default) |
| `noaa_json`  | last ~7 days                    | SWPC operational feed; `observed_flux`, narrowest feed auto-picked |
| `netcdf_url` | pre-2010 / reprocessed archives | pass `url=` (NOAA NCEI GOES science data) — *not yet implemented* |
| `local`      | anything                        | pass `path=` to a local netCDF/CSV — *not yet implemented* |

```python
xrap.fetch_goes_flares()                       # xray-flares-7-day: all events
xrap.fetch_goes_flares(latest=True)            # xray-flares-latest: most recent
xrap.fetch_goes_flares(start="2024-05-10", end="2024-05-15")   # filter by max_time

# force a specific SWPC feed instead of the auto-picked narrowest one
xrap.fetch_goes_xrs(source="noaa_json", feed="1-day")     # 6-hour | 1-day | 3-day | 7-day
```

Quick-look plot (needs `xrap[plot]`):

```python
import matplotlib.pyplot as plt
ds = xrap.fetch_goes_xrs("2017-09-06T09:00", "2017-09-06T15:00", source="sunpy", satellite=16)
xrap.plot_xrs(ds)          # flux (log y) vs time, with A/B/C/M/X guide lines

# overlay both absorption models on one axis
ax = None
for name in ("xrap", "drap2"):
    df = xrap.AbsorptionModel(freq_mhz=10.0, model=name).predict(
        lat=20, lon=30, start=ds.time.values[0], end=ds.time.values[-1], xray=ds)
    ax = xrap.plot_absorption(df, ax=ax, label=name.upper())
plt.show()
```

## Command line

Installing the package puts an `xrap` executable on your `PATH` (from the
`[project.scripts]` entry in `pyproject.toml`). Check it:

```bash
xrap --help
xrap predict --help
xrap fetch-goes --help
```

If `xrap` is not found, either activate the env it was installed into
(`conda activate xrap`) or run it as a module: `python -m xrap.cli --help`.

### `xrap predict` — absorption time series at a point

Writes CSV with columns `time, sza_deg, xrsb, absorption_db` (to stdout, or to
`-o FILE`). Downloads the GOES flux it needs automatically.

```bash
# 2017-09-06 X9.3 flare, 10 MHz, dayside point — CSV + plot
xrap predict --lat 20 --lon 30 --freq 10 \
    --start 2017-09-06T09:00 --end 2017-09-06T15:00 \
    -o absorption.csv --plot absorption.png

# recent event straight from the SWPC near-real-time feed (last 7 days only)
xrap predict --lat 0 --lon 0 --freq 10 \
    --goes-source noaa_json --feed 1-day \
    --start 2026-09-08T15:00 --end 2026-09-09T14:00 --plot abs_24h.png

# print to stdout and pipe elsewhere
xrap predict --lat 52 --lon 13 --freq 8 \
    --start 2017-09-10T15:00 --end 2017-09-10T17:00 | head
```

| flag | default | meaning |
|------|---------|---------|
| `--lat`, `--lon` | *required* | absorbing-point latitude / longitude (deg, E-positive lon) |
| `--freq` | `10.0` | HF frequency (MHz) |
| `--start`, `--end` | *required* | UTC window (any ISO string pandas can parse) |
| `--model` | `xrap` | `xrap` (Fiori et al. 2022) or `drap2` |
| `--grazing` / `--no-grazing` | per-model | Chapman grazing correction near the terminator (alt 90 km, H 7 km); default on for `xrap`, off for `drap2` |
| `--goes-source` | `sunpy` | `sunpy` (≥2010, science) or `noaa_json` (≤7 days) |
| `--satellite` | auto | e.g. `16` or `GOES-16` |
| `--feed` | auto | `noaa_json` only: `6-hour` / `1-day` / `3-day` / `7-day` |
| `--sza-method` | `noaa` | `noaa` (analytic) or `astropy` (ephemeris) |
| `--path` | `oneway` | `oneway` or `total` (round-trip) absorption |
| `-o`, `--output` | stdout | write CSV to this path |
| `--plot PNG` | — | also save an absorption-vs-time figure |

### `xrap fetch-goes` — GOES/XRS flux only

Writes netCDF if the output name ends in `.nc`, otherwise CSV.

```bash
xrap fetch-goes --start 2017-09-06T11:00 --end 2017-09-06T13:00 -o xrs.nc
xrap fetch-goes --source noaa_json --feed 1-day --plot xrs.png -o xrs.csv
```

| flag | default | meaning |
|------|---------|---------|
| `--start`, `--end` | required (optional with `--feed`) | UTC window |
| `--source` | `sunpy` | `sunpy` or `noaa_json` |
| `--satellite` | auto | e.g. `16` |
| `--resolution` | `1min` | `sunpy` only: `1min` (`avg1m`) or `1s` (`flx1s`) |
| `--feed` | auto | `noaa_json` feed; with it, `--start`/`--end` may be omitted |
| `-o`, `--output` | stdout | `.nc` → netCDF, else CSV |
| `--plot PNG` | — | also save a flux-vs-time figure |

## Tests & coverage

```bash
make test                    # pytest + coverage (config in pyproject.toml)
make coverage                # + HTML report -> htmlcov/index.html
```

Equivalently, with the env active:

```bash
pytest                       # fast suite, no network   (~3 s)
pytest --run-network         # + tests that download real GOES / SWPC data
pytest -q tests/test_cli.py  # just the CLI tests
pytest -k absorption         # filter by name
```

- The default suite is fully offline — HTTP is mocked and SunPy is not called.
- Tests marked `@pytest.mark.network` (live GOES/SWPC downloads + plots) are
  **skipped unless** you pass `--run-network`.
- Coverage prints a terminal summary and writes `coverage.xml`; `make coverage`
  also writes a browsable `htmlcov/` report.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `ruff` + the
offline `pytest` suite on Python 3.10–3.12 for every push and PR.

## Layout

```
src/xrap/
  __init__.py      public API
  sza.py           SZA from lat/lon/time (astropy + NOAA analytic)
  goes.py          GOES/XRS retrieval (sunpy / noaa_json / netcdf_url / local)
  absorption.py    absorption models: "xrap" (Fiori 2022) + "drap2"
  model.py         AbsorptionModel — end-to-end orchestration
  plot.py          plot_xrs / plot_absorption — quick-look figures (needs matplotlib)
  cli.py           `xrap` command line
  data/            bundled coefficients / lookup tables
tests/             pytest suite + fixtures (synthetic GOES flare);
                   `network`-marked tests download real data (`pytest --run-network`)
```

## References

- Fiori, R. A. D., Chakraborty, S., & Nikitina, L. (2022). Data-based
  optimization of a simple shortwave fadeout absorption model. *Journal of
  Atmospheric and Solar-Terrestrial Physics*, 230, 105843.
  [doi:10.1016/j.jastp.2022.105843](https://doi.org/10.1016/j.jastp.2022.105843)
- NOAA SWPC, *Global D-Region Absorption Prediction (D-RAP) Documentation*.
  <https://www.spaceweather.gov/content/global-d-region-absorption-prediction-documentation>

## License

MIT — see [LICENSE](LICENSE).
