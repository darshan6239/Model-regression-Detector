"""
Drift detection: tracks a rolling average of pass rates across the last
N runs. Catches gradual degradation that a single run-over-run diff
would miss (e.g. slowly declining model quality, or accumulating small
regressions from several small prompt tweaks over time).
"""
from src.db import get_recent_runs
from src.comparator import load_thresholds


def check_drift(prompt_version: str = None) -> dict:
    """
    Computes the rolling average pass rate over the last N runs and
    compares it to the historical average before that window.
    """
    thresholds = load_thresholds()
    window = thresholds["drift_window_runs"]
    drift_threshold_pct = thresholds["drift_threshold_pct"]

    all_runs = get_recent_runs(limit=window * 3)

    if prompt_version:
        all_runs = [r for r in all_runs if r["prompt_version"] == prompt_version]

    # Runs come back newest-first; reverse to chronological order.
    all_runs = list(reversed(all_runs))

    if len(all_runs) < window + 1:
        return {
            "drift_detected": False,
            "message": f"Not enough run history yet ({len(all_runs)} runs). "
                       f"Need at least {window + 1} to check drift.",
        }

    recent_window = all_runs[-window:]
    historical = all_runs[:-window]

    recent_avg = sum(r["aggregate_stats"]["category_pass_rate"] for r in recent_window) / len(recent_window)

    if not historical:
        return {
            "drift_detected": False,
            "message": "No historical runs before the current window to compare against yet.",
            "recent_window_avg": round(recent_avg, 4),
        }

    historical_avg = sum(r["aggregate_stats"]["category_pass_rate"] for r in historical) / len(historical)

    drift_pct = (historical_avg - recent_avg) * 100
    drift_detected = drift_pct >= drift_threshold_pct

    return {
        "drift_detected": drift_detected,
        "recent_window_avg": round(recent_avg, 4),
        "historical_avg": round(historical_avg, 4),
        "drift_pct": round(drift_pct, 2),
        "window_size": window,
        "message": (
            f"Slow drift detected: {window}-run rolling average pass rate "
            f"({recent_avg:.1%}) has dropped {drift_pct:.1f} points below the "
            f"historical average ({historical_avg:.1%})."
            if drift_detected else
            f"No significant drift. Rolling average: {recent_avg:.1%}, "
            f"historical average: {historical_avg:.1%}."
        ),
    }
