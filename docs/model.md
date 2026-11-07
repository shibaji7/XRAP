# XRAP model notes

Common inputs: `phi` = GOES 0.1-0.8 nm flux (W m^-2) **on the NOAA flare-class
scale**, `chi` = solar zenith angle at the absorbing point (deg), `f` = HF
frequency (MHz). Both models return **one-way** vertical absorption;
`path="total"` doubles it. `A >= 0`.

### Flux scale (read this first)

Both models are calibrated on the **flare-class scale**, where `X9.3 = 9.3e-4
W m^-2`. The reprocessed science-quality GOES L2 data that SunPy returns carry
the true irradiance, which is `1/0.7` larger in the long band: the 2017-09-06
flare reads 1.34e-3 on GOES-15 (GOES-16: 1.46e-3), and `1.3372e-3 x 0.7 =
9.36e-4 = X9.3`. The SWPC operational feeds (`noaa_json`) are already on the
class scale (`observed_flux` matches the flare-class value to ~0.1% across
recent flares).

So `fetch_goes_xrs()` multiplies the science-quality `xrsb` by
`FLARE_SCALE_XRSB = 0.7` by default (`flare_scale=True`; CLI `--flare-scale /
--no-flare-scale`) and leaves `noaa_json` alone. `ds.attrs["flux_scale"]` records
which you have (`"flare-class"` / `"science"`). If you pass your own science-quality
flux to `absorption()`, multiply by 0.7 first. Feeding raw GOES-16 science flux
into XRAP overstates absorption by `1/0.7 = 1.43x` (linear in flux).

**Data quality:** the L2 files carry `xrsb_quality` flags, which are returned but
**not** filtered. E.g. GOES-15 flatlines at ~7e-10 (flag 2) through 09:00-09:29 UT
on 2017-09-06 -- the X2.2 flare is missing there, so the figures below use the
clean 11:00-15:00 window.

`absorption(f, chi, phi, model="xrap"|"drap2", ...)` dispatches;
`AbsorptionModel(model=...)` and `xrap predict --model ...` follow.

## "xrap" — Fiori, Chakraborty & Nikitina (2022)   [IMPLEMENTED, DEFAULT]

doi:10.1016/j.jastp.2022.105843 — data-optimized fit to the NRCan 30 MHz
riometer network.

    A(f0) = C * phi^p_I * g(chi)^p_c        # dB, one-way, f0 = 30 MHz
    A(f)  = (f0 / f)^n * A(f0)              # f >= f_s;  2 f0^n/(f^n + f_s^n) * A(f0) below f_s

`f_s = freq_saturation_mhz` (default 5 MHz; see *Low-frequency saturation* under
[Oblique-path absorption](#oblique-path-absorption)). `C = 12080`, `p_I = 1`, `p_c = 1`, `n = 1.5` (Hargreaves; Parthasarathy et al.
1963). `reference_absorption(chi, phi)` returns `A(f0)`. The exponents are in
`src/xrap/data/xrap.json` (`scale`, `flux_exponent`, `cos_exponent`, `f0_mhz`,
`freq_exponent`, `freq_saturation_mhz`) so the paper's optimized coefficient sets drop in via
`coeffs=`.

## "drap2" — NOAA Global D-Region Absorption Prediction v2   [IMPLEMENTED]

https://www.spaceweather.gov/content/global-d-region-absorption-prediction-documentation

    HAF0      = a * log10(phi) + b          # MHz, sub-solar   (a = 10, b = 65)
    HAF(chi)  = HAF0 * (cos chi)^p          # p = 0.75; 0 for chi >= 90 deg
    A(f,chi)  = 0.5 * A_ref * (HAF(chi) / f)^n   # dB, one-way  (n = 1.5, A_ref = 1 dB)

`highest_affected_frequency(phi, chi)` returns `HAF(chi)`. Coefficients in
`src/xrap/data/drap2.json` (`haf_threshold_db = 10.0` for the polar-map HAF
definition).

## Model comparison — XRAP vs DRAP2

The two models are fit to different data and have different functional forms, so
they diverge — and the divergence depends almost entirely on **flare magnitude**.

### Response to flare size

![XRAP vs DRAP2 absorption vs flare magnitude](images/models_vs_flux.png)

*One-way absorption, overhead Sun, at 10 and 30 MHz, versus GOES 0.1–0.8 nm
flux. Dashed = 10 MHz, solid = 30 MHz; green = XRAP, blue = DRAP2.*

- **DRAP2 saturates.** Its flux dependence enters as `(a·log10(phi) + b)^1.5`.
  The `log10` compresses the dynamic range badly: going from X1 (10⁻⁴) to X10
  (10⁻³) raises `HAF0` only from 25 → 35 MHz, so absorption grows by just
  `(35/25)^1.5 ≈ 1.6×` for a **10×** increase in flux. There is also a hard
  floor: `HAF0 ≤ 0` (hence `A = 0`) for `phi ≲ 3×10⁻⁷` W m⁻² — the near-vertical
  blue cliff.
- **XRAP is linear in flux** (`A ∝ phi`), a straight line on the log–log plot.

Because both models share the `(f0/f)^1.5` frequency law and a near-identical
`cos χ` dependence, the **XRAP / DRAP2 ratio is essentially frequency- and
SZA-independent — it is a function of flare class**:

| flare | C1 | M1 | ~M2 | M5 | X1 | X10 |
|-------|----|----|-----|----|----|-----|
| XRAP / DRAP2 | 0.36 | 0.68 | **1.0** (crossover) | 1.9 | 3.2 | 19 |

So for weak events (C to ~M2) DRAP2 predicts *more* absorption; above ~M2 XRAP
predicts more, and by X-class DRAP2 is low by a factor of several to ~20×.

### Two real events

![GOES XRS, 2017-09-06](images/goes_xrs_2017-09-06.png)

![10 MHz absorption, 2017-09-06 flares](images/absorption_2017-09-06_10MHz_models.png)

The 2017-09-06 **X9.3** flare (GOES-15, class scale, peak `9.36e-4` = X9.36) at
10 MHz, 20°N 30°E: XRAP peaks near **49 dB**, DRAP2 near **2.7 dB** one-way.
49 dB is a complete HF blackout — consistent with what was observed (HF comms
were dead for ~1 h) — whereas DRAP2's 2.7 dB would imply usable HF throughout.
This is the underestimate Fiori et al. (2022) set out to fix by refitting to
riometer data. (Hand check at 33.0° SZA: `12080 x 9.36e-4 x cos(33.0°) x
(30/10)^1.5 = 49.3 dB`; the code gives 49.35 with its default grazing factor.)

![10 MHz absorption, last 24 h](images/absorption_last24h_10MHz_models.png)

A quiet day of **C-class** flares (0°N 0°E): here the ranking flips — DRAP2
(blue) sits *above* XRAP, consistent with the `C1` column of the table.

### Caveats for XRAP

- `C = 12080` and the linear-in-flux form were fit to **30 MHz** Canadian
  riometers over a limited set of events. Extrapolating far in frequency (the
  `(30/10)^1.5 ≈ 5.2×` step to 10 MHz) or to extreme flares carries real
  uncertainty; the paper reports the *simple* model can also over-predict (~1%
  high for an X2.1 in one event study), and the per-event optimized coefficients
  do better. Drop those in via `coeffs=` / `xrap.json`.
- DRAP2 remains useful as the operational reference and for weak events; keep it
  a call away with `model="drap2"`.

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

## Oblique-path absorption

`absorption()` is vertical. For a real tx-rx circuit (oblique incidence, 1+
hops), `src/xrap/oblique.py` -- `oblique_absorption(tx, rx, time, freq_mhz,
xray_wm2, n_hops=1, ...)` -- sums the vertical absorption over every D-region
crossing of the path, each at its own SZA, multiplied by the path-length
secant `sec(chi_D)` where `chi_D` is the incidence angle **at the D-region
height** (not at the F-layer reflection height) and the frequency is the actual
wave frequency (non-deviative absorption; same structure as ITU-R P.533-14
eq. 20). Two independent, user-selectable choices: `earth_model="curved"|"flat"`
(hop geometry) and `height_model="fixed"|"parabolic"` (F-layer reflection
height).

**Full derivation, citations, and a worked/validated example: see
[oblique.md](oblique.md).**

**Low-frequency saturation:** the `f^-1.5` law diverges toward 0 Hz, so XRAP
saturates it below `freq_saturation_mhz` (default 5 MHz; `S(f) = 2 f0^n /
(f^n + fs^n)` below the corner, continuous, power law unchanged above, bounded by
`2 (30/5)^1.5 = 29.4`). It is user-selectable (`freq_saturation_mhz=` on
`absorption()` / `oblique_absorption()` / `AbsorptionModel`, `--freq-saturation
MHZ|none` on the CLI, or `coeffs`); `None` disables it. DRAP2 keeps its published,
unsaturated form unless you opt in. Derivation and numbers: [oblique.md §5.1](oblique.md).

**`max_db` (peak scaling):** saturation removes the divergence toward low
frequency, but the models still don't saturate in flux (XRAP is linear), so a
strong flare on a low-frequency or long multi-hop path can reach thousands of dB --
e.g. at 80 m (3.75 MHz) a *vertical* X9 crossing is ~194 dB (246 dB without
saturation). Those numbers mean "blackout," not a literal margin, and the models
are not validated that far from their 30 MHz calibration. Pass `max_db=` to
`absorption()` / `oblique_absorption()`: the **whole returned series is multiplied
by one factor so its peak equals `max_db`** (only when the peak exceeds it; shape
preserved, no flat top; `xrap.peak_scale()` returns the factor). The factor depends
on the window, quiet values are rescaled too, and separately-scaled series are not
comparable; `max_db_mode="clip"` gives the old hard cut (`result >= max_db` =
blackout). Details: [oblique.md §5.1](oblique.md). No SNR / link budget here --
this stops at absorption (dB).

## Out of scope

- Solar-energetic-proton / polar-cap term — `polar_cap_absorption()` is a
  documented `NotImplementedError` stub.
- SNR / link budget (FSPL, antenna gain, noise floor) — `oblique_absorption()`
  stops at absorption in dB.
- Full ray tracing (Croft-Hoogasian, magnetoionic splitting, deviative
  absorption) — the secant law + quasi-parabolic layer is the standard
  simplified treatment, not a substitute for a full ray tracer.

## CLI (`src/xrap/cli.py`)  [DONE]

- `xrap predict --lat --lon --freq --start --end` -> absorption CSV (+ `--plot`);
  flags `--model xrap|drap2`, `--grazing/--no-grazing`, `--goes-source`,
  `--satellite`, `--feed`, `--flare-scale/--no-flare-scale`, `--sza-method`,
  `--path`, `--freq-saturation MHZ|none`.
- `xrap fetch-goes --start --end` -> `.nc` (netCDF) or CSV (+ `--plot`); flags
  `--source`, `--satellite`, `--resolution`, `--feed`,
  `--flare-scale/--no-flare-scale`.
- `max_db` / `oblique_absorption` are Python-API only (not exposed on the CLI or
  `AbsorptionModel`).

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

## Figures

The plots in this document are regenerated by `docs/make_figures.py` (needs
network + matplotlib) into `docs/images/`. The "last 24 h" panels change each
run. [oblique.md](oblique.md)'s figure is regenerated separately by
`docs/make_oblique_figures.py`.
