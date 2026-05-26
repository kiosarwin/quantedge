from rich.console import Console

from src.dashboard import build_cohort_expectancy


def test_build_cohort_expectancy_ranks_by_expectancy():
    state = {
        "lifecycle": {
            "cohorts": [
                {"key": "cohort-A", "trades": 12, "win_rate": 0.58, "profit_factor": 1.30, "expectancy_usd": 2.40, "status": "ACTIVE"},
                {"key": "cohort-B", "trades": 20, "win_rate": 0.62, "profit_factor": 1.10, "expectancy_usd": 1.10, "status": "PAPER_VALIDATION"},
                {"key": "cohort-C", "trades": 6, "win_rate": 0.50, "profit_factor": 1.80, "expectancy_usd": 3.20, "status": "RESEARCH"},
            ]
        }
    }

    panel = build_cohort_expectancy(state)
    console = Console(record=True, width=120)
    console.print(panel)
    text = console.export_text()

    assert "Cohort Expectancy" in text
    assert text.index("cohort-C") < text.index("cohort-A") < text.index("cohort-B")
    assert "ACTIVE" in text
    assert "PAPER_VALIDATION" in text
