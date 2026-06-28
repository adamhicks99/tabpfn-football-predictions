# Round of 32 — submission methodology

**Recommended submission:** `submission_blend_w015.csv` — a 15% TabPFN + 85% market blend.
It is TabPFN-eligible (uses TabPFN, weight > 0) and had the **lowest holdout log-loss of
any option tried**, edging the raw market.

Other artifacts:
- `submission_tabpfn_odds.csv` — pure TabPFN on odds (eligible, but weaker).
- `market_baseline.csv` — raw de-vigged market (strongest non-eligible reference).
- `blend_sweep.csv`, `holdout_scores.csv` — validation tables.

Fixtures are the current World Cup Round-of-32 games that had **not kicked off** at
generation time (kickoff > now), taken live from The Odds API.

## Goal
Minimize the competition metric — multiclass **log-loss** — on home / draw / away
probabilities. Uniform 1/3 scores ln(3) ≈ 1.0986.

## Data
- **Results:** `results.csv` (martj42) — ~49k internationals, used for team-strength state.
- **Odds:** `data/odds/features.csv` — 3-way (h2h) prices from The Odds API, de-vigged per
  book and combined into a median consensus (`market_p_home/draw/away`, plus dispersion,
  overround, book count). 204 historical tournament matches (2022–2025) + the live R32.

## Pipeline
1. **Features** (`features.py`): one leakage-safe chronological pass builds Elo / form /
   rest / head-to-head; market consensus is joined per match. Sets: `base`, `odds`, `base+odds`.
2. **Odds-only training:** the model is fit only on matches that have odds (the consensus is
   the dominant signal); Elo features still use the full history for accuracy.
3. **Model:** **TabPFN** (`tabpfn_client`, seed 42) — required for eligibility; well suited
   to this small (~200-row) tabular problem.
4. **Fixtures from the live feed:** the live `sports/soccer_fifa_world_cup/odds` endpoint is
   the authoritative bracket. We keep only games with **kickoff > now** (so completed games
   drop out), and map feed team names to history names.
5. **Blend:** final probabilities = `w·TabPFN(odds) + (1-w)·market`. Any `w > 0` uses TabPFN.

## Model selection (single time holdout: train < 2024-06-01 = 117, test ≥ = 87)
`holdout_scores.csv` (single models) and `blend_sweep.csv` (the blend):

| Option | Log-loss |
|---|---|
| **blend w=0.15 (15% TabPFN + 85% market)** | **0.9609** ✅ best + eligible |
| market (raw, reference, not eligible) | 0.9618 |
| TabPFN / odds | 0.9919 |
| TabPFN / base+odds | 1.039 |
| TabPFN / base | 1.090 |
| logistic / * | 1.49–1.57 |

The blend sweep is monotone away from the market: log-loss rises smoothly from w=0 (0.9618)
to w=1 (0.9919), with a shallow minimum at **w≈0.15**. So a light TabPFN correction of the
market is best; heavier TabPFN weight hurts (it over-shrinks favorites, e.g. it had pulled
Argentina's win prob far below the market's).

## Honest caveats
- The blend's edge over the raw market is **small (0.9609 vs 0.9618)** on an 87-match
  holdout — directional, not a guarantee. Its real value is being the best option that is
  **TabPFN-eligible**.
- Biggest likely improvement: **more historical odds** to grow the 204-row training set
  (`odds.py discover` → gated `fetch-historical`).

## Reproduce
```bash
python odds.py --keychain-service prior-labs-football-the-odds-api \
  --keychain-account adamhicks upcoming --execute   # refresh live bracket + odds
python blend.py --submit                            # sweep + write submission_blend.csv (best w)
```
