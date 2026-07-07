"""
Comparison logic: diffs the current run against the previous baseline run.
This is the core value of the whole system.
"""
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "thresholds.yaml"


def load_thresholds() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def compare_runs(current_results: list[dict], current_stats: dict,
                  baseline_results: list[dict] | None, baseline_stats: dict | None) -> dict:
    """
    Compares the current run to the baseline run and returns a structured
    diff: overall pass rate delta, per-category deltas, regressions, and
    improvements, plus a severity classification.
    """
    thresholds = load_thresholds()

    if baseline_results is None:
        # First-ever run — nothing to compare against.
        return {
            "has_baseline": False,
            "severity": "info",
            "message": "This is the first recorded run. No baseline to compare against.",
            "overall_pass_rate_delta": None,
            "regressions": [],
            "improvements": [],
        }

    baseline_by_id = {r["test_case_id"]: r for r in baseline_results}

    regressions = []
    improvements = []

    for current in current_results:
        tc_id = current["test_case_id"]
        baseline = baseline_by_id.get(tc_id)
        if baseline is None:
            continue  # New test case with no baseline counterpart, skip diffing it

        was_pass = baseline["category_match"]
        is_pass = current["category_match"]

        if was_pass and not is_pass:
            regressions.append({
                "test_case_id": tc_id,
                "input_text": current["input_text"],
                "expected_category": current["expected_category"],
                "baseline_actual": baseline["actual_category"],
                "current_actual": current["actual_category"],
                "difficulty": current["difficulty"],
            })
        elif not was_pass and is_pass:
            improvements.append({
                "test_case_id": tc_id,
                "input_text": current["input_text"],
                "expected_category": current["expected_category"],
                "baseline_actual": baseline["actual_category"],
                "current_actual": current["actual_category"],
                "difficulty": current["difficulty"],
            })

    overall_delta = current_stats["category_pass_rate"] - baseline_stats["category_pass_rate"]

    # Per-category deltas
    per_category_delta = {}
    for category in ["billing", "technical", "account", "general"]:
        current_cat_results = [r for r in current_results if r["expected_category"] == category]
        baseline_cat_results = [r for r in baseline_results if r["expected_category"] == category]
        if current_cat_results and baseline_cat_results:
            current_rate = sum(1 for r in current_cat_results if r["category_match"]) / len(current_cat_results)
            baseline_rate = sum(1 for r in baseline_cat_results if r["category_match"]) / len(baseline_cat_results)
            per_category_delta[category] = round(current_rate - baseline_rate, 4)

    # Severity classification based on configurable thresholds
    abs_delta_pct = abs(overall_delta) * 100
    if overall_delta < 0 and abs_delta_pct >= thresholds["critical_threshold_pct"]:
        severity = "critical"
    elif overall_delta < 0 and abs_delta_pct >= thresholds["warning_threshold_pct"]:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "has_baseline": True,
        "severity": severity,
        "overall_pass_rate_delta": round(overall_delta, 4),
        "per_category_delta": per_category_delta,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }
