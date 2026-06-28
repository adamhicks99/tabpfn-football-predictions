# Round of 32 — submission methodology

**Submission file:** `submission_tabpfn_odds.csv` (model) — eligible because it uses TabPFN.
**Baseline (reference):** `market_baseline.csv` — raw de-vigged bookmaker consensus.
**Generated:** 2026-06-27 (UTC). 16 fixtures, the full 2026 World Cup Round of 32.

## Goal
Predict home-win / draw / away-win probabilities for each match and minimize the
competition metric: **multiclass log-loss** (mean negative log-probability of the actual
outcome). Lower is better; the uniform 1/3-each baseline scores ln(3) ≈ 1.0986.

## Data
- **Match results:** `results.csv` (martj42/international_results) — ~49k internationals,
  used to compute team strength state (Elo, recent form, head-to-head, rest).
- **Betting odds:** `data/odds/features.csv` — historical 3-way (h2h) prices from The Odds
  API, de-vigged per bookmaker and combined into a robust median consensus
  (`market_p_home/draw/away`, plus dispersion, overround, and book count). 204 historical
  tournament matches (2022–2025) plus the live Round-of-32 fixtures.

## Pipeline
1. **Features** (`features.py`): one leakage-safe chronological pass builds Elo, form,
   rest, and head-to-head features using only prior matches; market consensus columns are
   joined per match. Feature sets: `base` (26 engineered), `odds` (market consensus only),
   `base+odds`.
2. **Odds-only training universe:** training is **restricted to matches that have odds**
   (the de-vigged consensus is the dominant signal; engineered features cover only ~the
   same tournament universe). Team-strength features are still computed from the full
   history for accuracy, but the model is fit only on odds-covered rows.
3. **Model:** **TabPFN** (`tabpfn_client`, `ignore_pretraining_limits=True`, seed 42) — a
   foundation model well suited to this small (hundreds of rows) tabular problem, and
   required for an eligible submission.
4. **Fixtures from the live feed:** the Round-of-32 bracket and its odds are taken from The
   Odds API live `sports/soccer_fifa_world_cup/odds` endpoint — the authoritative source of
   which games are happening (the static results file had stale pairings). Feed team names
   are mapped to history names so fixtures inherit the right team state.
5. **Prediction:** fixtures are appended to the history, features are built, and TabPFN
   (fit on all 204 odds-covered matches) predicts each fixture. Output is the exact Prior
   schema: `date,home_team,away_team,p_home_win,p_draw,p_away_win`, probabilities normalized
   to sum to one.

## Model selection (validation)
Single time-based holdout on the odds-covered matches: train before `2024-06-01`
(117 matches), score on/after it (87 matches). Full table: `holdout_scores.csv`.

| Model | Features | Log-loss | Accuracy |
|---|---|---|---|
| **TabPFN** | **odds** | **0.992** | **54.0%** |
| TabPFN | base+odds | 1.039 | 44.8% |
| TabPFN | base | 1.090 | 37.9% |
| logistic | base / base+odds | 1.49 / 1.57 | — |
| *market (reference)* | *raw de-vigged* | *0.962* | — |

**Chosen:** TabPFN + `odds` — the best-scoring eligible (TabPFN) configuration. It learns a
light recalibration of the bookmaker consensus (e.g. it shrinks extreme prices toward the
center).

## Honest caveats
- The **raw market (0.962) still edges every trained model** on this holdout, so
  `market_baseline.csv` is the strongest single set of probabilities — but it does not use
  TabPFN, so it is not eligible. `submission_tabpfn_odds.csv` is the best **eligible** option.
- Validation is a **single small holdout** (87 tournament matches); treat the ranking as
  directional, not precise.
- Largest expected improvement: **more historical odds** to enlarge the training set
  (`odds.py discover` → gated `fetch-historical`), and a TabPFN×market blend.

## Reproduce
```bash
python odds.py --keychain-service prior-labs-football-the-odds-api \
  --keychain-account adamhicks upcoming --execute   # refresh fixtures + odds
python predict.py --model tabpfn --features odds     # writes submission.csv
```
