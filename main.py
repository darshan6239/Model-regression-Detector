"""
Main CLI entrypoint for the Model Regression Detection pipeline.

Usage:
    python main.py --prompt-version v1
    python main.py --prompt-version v2 --no-slack
"""
import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone

from src.runner import run_eval, load_golden_dataset
from src.scorer import compute_aggregate_stats
from src.comparator import compare_runs
from src.drift_tracker import check_drift
from src.db import save_run, get_latest_run
from reports.report_generator import generate_report
from alerts.slack_notifier import send_slack_alert


async def main(prompt_version: str, send_slack: bool = True):
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # 1. Run the golden dataset through the classifier
    results = await run_eval(prompt_version)
    stats = compute_aggregate_stats(results)

    dataset = load_golden_dataset()
    dataset_version = dataset["dataset_version"]

    # 2. Fetch the previous run as baseline (before saving the current one)
    baseline_run = get_latest_run()
    baseline_results = baseline_run["results"] if baseline_run else None
    baseline_stats = baseline_run["aggregate_stats"] if baseline_run else None

    # 3. Diff current vs baseline
    diff = compare_runs(results, stats, baseline_results, baseline_stats)

    # 4. Save this run to history
    from src.classifier import load_prompt_config
    prompt_config = load_prompt_config(prompt_version)
    save_run(
        run_id=run_id,
        prompt_version=prompt_version,
        model=prompt_config.get("model", "unknown"),
        dataset_version=dataset_version,
        aggregate_stats=stats,
        results=results,
    )

    # 5. Check for slow drift across recent runs
    drift = check_drift(prompt_version=prompt_version)

    # 6. Generate HTML report
    run_metadata = {
        "run_id": run_id,
        "prompt_version": prompt_version,
        "model": prompt_config.get("model", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
    }
    report_path = generate_report(run_metadata, stats, results, diff)

    # 7. Send Slack alert
    if send_slack:
        send_slack_alert(run_metadata, stats, diff, report_url=f"file://{report_path}")

    # 8. Print summary to console (useful for CI logs)
    print("\n" + "=" * 60)
    print(f"RUN COMPLETE: {run_id}")
    print(f"Prompt version: {prompt_version}")
    print(f"Pass rate: {stats['category_pass_rate']:.1%}")
    print(f"Severity: {diff['severity']}")
    if diff.get("has_baseline"):
        print(f"Delta vs baseline: {diff['overall_pass_rate_delta']:+.1%}")
        print(f"Regressions: {diff['regression_count']} | Improvements: {diff['improvement_count']}")
    print(f"Drift check: {drift['message']}")
    print(f"Report: {report_path}")
    print("=" * 60)

    # 9. Exit non-zero if critical, so CI can block merge
    if diff["severity"] == "critical":
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model regression eval pipeline.")
    parser.add_argument("--prompt-version", default="v1", help="Prompt version to test, e.g. v1 or v2")
    parser.add_argument("--no-slack", action="store_true", help="Skip sending the Slack alert")
    args = parser.parse_args()

    asyncio.run(main(args.prompt_version, send_slack=not args.no_slack))
