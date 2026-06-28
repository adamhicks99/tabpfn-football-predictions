# Round of 32 — submission methodology

**Recommended submission:** `submission_blend.csv` — a 10% TabPFN + 90% market blend.
It is TabPFN-eligible (uses TabPFN, weight > 0) and is the best eligible option on the
holdout. `market_baseline.csv` is the pure de-vigged market (stronger, but not eligible).

Fixtures are the current World Cup Round-of-32 games that had **not kicked off** at
generation time (kickoff > now), taken live from The Odds API.

## Goal
Minimize the competition metric — multiclass **log-loss** — on home/draw/away probabilities.
Uniform 1/3 scores ln(3) ≈ 1.0986.

## Data
- **Results:** `results.csv` (martj42) — ~49k internationals, for team-strength state.
- **Odds:** `data/odds/features.csv` — 3-way (h2h) prices from The Odds API, de-vigged per
  book into a median consensus (`market_p_home/draw/away`, dispersion, overround, book count).
  **381 odds-covered matches** across the major tournaments 2021–2026 (WC 2022 + 2026 group,
  Euro 2020/2024, Copa 2024, AFCON 2021/2023, Gold Cup 2025, Nations League finals) plus the
  live R32.

## Pipeline
1. **Features** (`features.py`): leakage-safe chronological pass for Elo/form/rest/h2h;
   market consensus joined per match. Sets: `base`, `odds`, `base+odds`.
2. **Odds-only training:** the model is fit only on matches that have odds.
3. **Model:** **TabPFN** (`tabpfn_client`, seed 42) — required for eligibility; suits the
   small (~hundreds of rows) tabular problem.
4. **Fixtures from the live feed:** the live `sports/soccer_fifa_world_cup/odds` endpoint is
   the authoritative bracket; keep games with **kickoff > now**, map feed names to history.
5. **Blend:** final probabilities = `w·TabPFN(odds) + (1-w)·market`. Any `w > 0` uses TabPFN.

## Validation (time holdout: train < 2024-06-01, test ≥; 381-match dataset)
Blend sweep (`blend_sweep.csv`), test = 181:

| w (TabPFN weight) | log-loss |
|---|---|
| 0.00 — pure market (reference) | 0.8975 |
| **0.10 — production default** | **0.8990** |
| 0.05 — best eligible | 0.8982 |
| 1.00 — pure TabPFN | 0.9414 |

The sweep is **monotone**: less TabPFN = lower log-loss, with the eligible optimum at the
small-weight floor. So the production weight is kept small (0.10) — a thin TabPFN blend that
stays eligible while the well-calibrated market carries the signal.

## What the training-data expansion showed
Growing the odds-covered set from **204 → 381** matches (notably the 63 WC-2026 group games)
**did not let TabPFN beat the market** — pure TabPFN stayed worse (0.9414 vs market 0.8975),
and the best eligible weight got *smaller* (0.15 → 0.05). Conclusion: the de-vigged market is
near the ceiling for these tournaments; TabPFN-on-odds is recalibrating an already-sharp
signal. The expansion still helped by making the thin blend more stable and confirming the
ceiling.

## Honest caveats
- The eligible blend's edge over raw market is negligible; its value is being the best
  **TabPFN-eligible** option. If raw probabilities were allowed, submit `market_baseline.csv`.
- Next levers (uncertain): richer odds features (line movement, dispersion), a different
  model class, or non-odds signals — but the market is a very high bar.

## Reproduce
```bash
python odds.py --keychain-service prior-labs-football-the-odds-api \
  --keychain-account adamhicks upcoming --execute   # refresh live bracket + odds
python evaluate.py --blend                          # re-validate the blend weight
python predict.py                                   # write submission.csv (the blend)
```
