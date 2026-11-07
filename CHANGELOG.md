# Changelog

Notable changes. The project has no tagged releases yet; everything below is
**Unreleased** relative to the last pushed commit (`f28ac01`, "Add
oblique/multi-hop absorption via the secant law").

## Unreleased

### Added (latest)

- `max_db` / `max_db_mode` on `AbsorptionModel` and `xrap predict --max-db DB
  --max-db-mode scale|clip`. When `max_db` is set, `predict()` and the CLI CSV also
  carry an `absorption_raw_db` column (peak scaling is irreversible); without it the
  output columns are unchanged.

### Fixed

- **Flux scale.** The models (XRAP's `12080`, DRAP2's HAF law) are calibrated on
  the NOAA flare-class scale (`X9.3 = 9.3e-4 W/m²`), but the SunPy science-quality
  GOES L2 data carry the true irradiance, ~1/0.7 larger in the long band
  (GOES-15 reads `1.337e-3` for the 2017-09-06 X9.3; `× 0.7 = 9.36e-4`).
  `fetch_goes_xrs()` now multiplies science-quality `xrsb` by
  `FLARE_SCALE_XRSB = 0.7` by default. The SWPC `noaa_json` route was already on
  the flare-class scale and is untouched. Previously absorption for a given event
  was ~1.43–1.57× too large (e.g. 77 dB instead of 49 dB at 10 MHz, 20°N 30°E for
  the X9.3 flare), because XRAP is linear in flux.
- **Oblique absorption, obliquity law.** `sec(χ)^(n+1)` (`sec^2.5`) double-counted
  the obliquity; the non-deviative D-region law is `sec(χ_D)` at the actual
  frequency, independent of `n` (same structure as ITU-R P.533-14 eq. 20).
- **Oblique absorption, incidence angle.** It was evaluated at the F-layer
  reflection height; it is now evaluated at the D-region height (curved-Earth
  Bouguer invariant), which is larger (e.g. 79.5° vs 72.3° on a 3000 km hop).
- **Oblique absorption, crossing positions.** The two D-region crossings of a hop
  are now located exactly on the straight chord instead of by a linear rule
  (~95 km off on a 2200 km hop).

Net effect on oblique totals: about 3–3.6× lower on 1500–3500 km hops. See
`docs/oblique.md` §7 for the full write-up.

### Added

- `freq_saturation_mhz` — continuous low-frequency saturation of the `f^-1.5` law
  (XRAP default 5 MHz; `None` disables; DRAP2 off unless requested). Available on
  `absorption()`, `oblique_absorption()`, `AbsorptionModel`, `scale_frequency(f_sat_mhz=)`,
  `coeffs`, and the CLI (`--freq-saturation MHZ|none`).
- `max_db` on `absorption()` / `oblique_absorption()` — **scales the whole returned
  series so its peak equals `max_db`** (down only; shape preserved), with
  `max_db_mode="clip"` for a hard cut; public helper `peak_scale()`.
- `crossing_geometry()`; `HopCrossing.mirror_incidence_deg`.
- `fetch_goes_xrs(flare_scale=...)`, `ds.attrs["flux_scale"]`, CLI
  `--flare-scale/--no-flare-scale`.
- Docs: `docs/oblique.md` (derivation, worked example, corrections),
  `docs/make_oblique_figures.py`, `docs/make_maxdb_figure.py`; flux-scale section in
  `docs/model.md`.

### Changed (breaking)

- `fetch_goes_xrs(source="sunpy")` returns class-scale `xrsb` by default. Pass
  `flare_scale=False` for the raw science flux (the previous behaviour).
- `obliquity_factor(incidence_deg, *, exponent=1.0)` — was `obliquity_factor(incidence_deg, *, n=1.5)`
  meaning `sec^(n+1)`.
- `oblique_absorption(..., obliquity_exponent=1.0)` replaces `freq_exponent=`.
- `HopCrossing.incidence_deg` is now the angle at the D-region height (the mirror-level
  angle is `mirror_incidence_deg`).
- XRAP values below 5 MHz are lower than before because of the saturation; pass
  `freq_saturation_mhz=None` to recover the previous unbounded law.
- Docs figures for the 2017-09-06 event now use GOES-15 (the satellite behind the
  official X9.3 class), 11:00–15:00 UT.

### Known gaps

- GOES `xrsb_quality` flags are returned but not filtered (e.g. GOES-15 flatlines
  at ~7e-10 through 09:00–09:29 UT on 2017-09-06).
- `oblique_absorption()` has no CLI command.
- `netcdf_url` / `local` GOES routes and `polar_cap_absorption()` are stubs.
- The 5 MHz saturation corner is a placeholder, not calibrated against data.

## `f28ac01` — oblique/multi-hop absorption (2026-09-14)

First version of `xrap.oblique` (superseded by the fixes above; see
`docs/oblique.md` §7).
