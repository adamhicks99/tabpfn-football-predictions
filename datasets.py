"""Certified evaluation datasets: named, version-controlled test protocols.

Every backtest score is only meaningful relative to one of these. They are
defined once here, referenced by name, and recorded on every experiment (along
with the data SHA) so results always trace back to a known test set. Scores are
only comparable within the same ``eval_dataset`` and data version.

Add one by adding an entry to ``EVAL_DATASETS``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Big international tournaments (knockout-heavy) -- closest to what the
# competition actually predicts.
TOURNAMENT_REGEX = (
    r"^(?:FIFA World Cup|UEFA Euro|Copa América|African Cup of Nations"
    r"|Gold Cup|UEFA Nations League)$"
)


@dataclass(frozen=True)
class EvalDataset:
    """A walk-forward evaluation protocol (the test side of a backtest)."""

    name: str
    description: str
    start: str
    train_start: str = "2014-01-01"
    end: str | None = None  # None -> latest played match + 1 day
    test_months: int = 3
    step_months: int = 3
    train_months: int | None = None  # None -> expanding window
    max_train_rows: int | None = 10_000
    min_train_rows: int = 1_000
    tournament_regex: str | None = None

    def as_params(self) -> dict[str, object]:
        """Flat dict of the spec for logging (so a run traces back to it)."""
        return asdict(self)


EVAL_DATASETS: dict[str, EvalDataset] = {
    "full_2018": EvalDataset(
        name="full_2018",
        description="All internationals, 2018->latest, 3-month walk-forward.",
        start="2018-01-01",
    ),
    "recent_3y": EvalDataset(
        name="recent_3y",
        description="All internationals, last ~3 years, 3-month walk-forward.",
        start="2023-06-01",
    ),
    "tournaments_2022": EvalDataset(
        name="tournaments_2022",
        description=(
            "Big tournaments only (World Cup/Euro/Copa/AFCON/Gold Cup/"
            "Nations League), 2022-11->2025-07."
        ),
        start="2022-11-01",
        end="2025-07-01",
        tournament_regex=TOURNAMENT_REGEX,
    ),
}


def get_dataset(name: str) -> EvalDataset:
    """Look up a certified evaluation dataset by name."""
    try:
        return EVAL_DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown eval dataset {name!r}. Available: {', '.join(EVAL_DATASETS)}"
        ) from None
