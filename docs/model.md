# XRAP model notes

Common inputs: `phi` = GOES 0.1-0.8 nm flux (W m^-2), `chi` = solar zenith angle
at the absorbing point (deg), `f` = HF frequency (MHz). Both models return
**one-way** vertical absorption; `path="total"` doubles it. `A >= 0`.

`absorption(f, chi, phi, model="xrap"|"drap2", ...)` dispatches;
`AbsorptionModel(model=...)` and `xrap predict --model ...` follow.

## "xrap" — Fiori, Chakraborty & Nikitina (2022)   [IMPLEMENTED, DEFAULT]

doi:10.1016/j.jastp.2022.105843 — data-optimized fit to the NRCan 30 MHz
riometer network.

    A(f0) = C * phi^p_I * g(chi)^p_c        # dB, one-way, f0 = 30 MHz
    A(f)  = (f0 / f)^n * A(f0)

`C = 12080`, `p_I = 1`, `p_c = 1`, `n = 1.5` (Hargreaves; Parthasarathy et al.
1963). `reference_absorption(chi, phi)` returns `A(f0)`. The exponents are in
`src/xrap/data/xrap.json` (`scale`, `flux_exponent`, `cos_exponent`, `f0_mhz`,
`freq_exponent`) so the paper's optimized coefficient sets drop in via
`coeffs=`.

## "drap2" — NOAA Global D-Region Absorption Prediction v2   [IMPLEMENTED]

https://www.spaceweather.gov/content/global-d-region-absorption-prediction-documentation

    HAF0      = a * log10(phi) + b          # MHz, sub-solar   (a = 10, b = 65)
    HAF(chi)  = HAF0 * (cos chi)^p          # p = 0.75; 0 for chi >= 90 deg
    A(f,chi)  = 0.5 * A_ref * (HAF(chi) / f)^n   # dB, one-way  (n = 1.5, A_ref = 1 dB)

`highest_affected_frequency(phi, chi)` returns `HAF(chi)`. Coefficients in
`src/xrap/data/drap2.json` (`haf_threshold_db = 10.0` for the polar-map HAF
definition).

## Zenith-angle handling (`g(chi)` / the `cos chi` taper)

Shared primitive `zenith_factor(chi, exponent=, grazing=, alt_km=90, H_km=7)`:

- `grazing=False` — `max(cos chi, 0)^exponent`, hard zero at `chi = 90 deg`.
- `grazing=True` — `(1 / Ch(X, chi))^exponent` where the layer is sunlit, else 0
  (`xrap.sza.chapman_function`, Smith & Smith 1972). A small positive value
  through 90 deg that decays smoothly to zero at `terminator_sza(alt_km)`
  (~99.6 deg at 90 km); equals `cos chi` below ~80 deg.

`grazing` default: **True for "xrap"** (`cos chi` there is the overhead
ionization-rate factor, for which `1/Ch` is the correct grazing generalization),
**False for "drap2"** (faithful to the published spec). `alt_km = 90`,
`H_km = 7` are hardcoded defaults, overridable per call.

## scale_frequency

`scale_frequency(A0, f0, f, n=None)` — standalone `A(f) = (f0/f)^n * A(f0)`,
`n` defaults to 1.5.

## Out of scope

- Solar-energetic-proton / polar-cap term — `polar_cap_absorption()` is a
  documented `NotImplementedError` stub.
- Oblique (slant-path) geometry — models are vertical; apply an obliquity factor
  externally.

## CLI (`src/xrap/cli.py`)  [DONE]

- `xrap predict --lat --lon --freq --start --end` -> absorption CSV (+ `--plot`);
  flags `--model xrap|drap2`, `--grazing/--no-grazing`, `--goes-source`,
  `--satellite`, `--feed`, `--sza-method`, `--path`.
- `xrap fetch-goes --start --end` -> `.nc` (netCDF) or CSV (+ `--plot`); flags
  `--source`, `--satellite`, `--resolution`, `--feed`.

## Plots (`src/xrap/plot.py`)  [DONE]

- `plot_xrs(ds)` — GOES flux, log y, A/B/C/M/X guides.
- `plot_absorption(df, ax=, label=, color=)` — absorption (dB) vs time from
  `AbsorptionModel.predict`; call repeatedly with one `ax` to overlay models
  (a legend appears when any call passes `label=`). The two `network` tests in
  `test_plot.py` overlay `"xrap"` vs `"drap2"` for the 2017-09 flare and the
  last 24 h.

## SZA (`src/xrap/sza.py`)  [IMPLEMENTED]

- `method="noaa"` (default) — analytic NOAA low-precision, vectorised.
- `method="astropy"` — full ephemeris; honours `alt_km` + `refraction`.
- Height helpers for grazing: `horizon_dip`, `terminator_sza`, `is_sunlit`,
  `chapman_function`.
- Open question: sub-ionospheric (ray-piercing) point vs. ground point.

## GOES retrieval (`src/xrap/goes.py`)

- `sunpy` [DONE] — `Fido` + `a.Instrument.xrs` + `a.Resolution("avg1m"|"flx1s")`
  + optional `a.goes.SatelliteNumber`; loads via `TimeSeries`, columns
  `xrsa`/`xrsb` (+ `*_quality`). Newest satellite chosen if unspecified.
  Science data ~2010 -> present.
- `noaa_json` [DONE] — `services.swpc.noaa.gov/json/goes/{primary,secondary}/`;
  reads **`observed_flux`** for both bands from the narrowest `xrays-*` feed
  covering the request, or an explicit `feed="6-hour"|"1-day"|"3-day"|"7-day"`
  (then `start`/`end` are optional). `fetch_goes_flares()` reads
  `xray-flares-{7-day,latest}.json`. <= 7 days only.
- `xrap.plot.plot_xrs(ds)` [DONE] — flux (log y) vs time, A/B/C/M/X guide lines;
  covered by `test_plot.py` (synthetic offline + 2017-09 flare & last-24 h
  `network` tests).
- `netcdf_url` [TODO] — NOAA NCEI GOES science-data netCDF for pre-2010 events.
- `local` [TODO] — user-supplied file per event.

Live routes are covered by `@pytest.mark.network` tests (`pytest --run-network`);
the default suite mocks `xrap.goes._get_json`.

## Calibration / validation datasets (TODO)

- riometer absorption during known M/X flares
- HF Doppler / SuperDARN blackout extents
- direct DRAP2 map comparison
