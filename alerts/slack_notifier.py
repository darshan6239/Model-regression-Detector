"""
Sends a formatted alert to Slack via incoming webhook whenever an eval
run completes. Free — just requires a Slack workspace + webhook URL.
Docs: https://api.slack.com/messaging/webhooks
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

SEVERITY_EMOJI = {
    "ok": "✅",
    "warning": "⚠️",
    "critical": "🚨",
    "info": "ℹ️",
}


def send_slack_alert(run_metadata: dict, stats: dict, diff: dict, report_url: str = None) -> bool:
    """
    Sends a Slack message summarizing the eval run. Returns True on success,
    False if the webhook isn't configured or the request fails.
    """
    if not SLACK_WEBHOOK_URL:
        print("[slack_notifier] SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return False

    severity = diff.get("severity", "info")
    emoji = SEVERITY_EMOJI.get(severity, "")

    pass_rate = stats["category_pass_rate"] * 100

    headline = f"{emoji} Eval run `{run_metadata['run_id']}` — prompt `{run_metadata['prompt_version']}` — {severity.upper()}"

    lines = [
        f"*Pass rate:* {pass_rate:.1f}%",
    ]

    if diff.get("has_baseline"):
        delta_pct = diff["overall_pass_rate_delta"] * 100
        delta_str = f"{delta_pct:+.1f}%"
        lines.append(f"*Change vs baseline:* {delta_str}")
        lines.append(f"*Regressions:* {diff['regression_count']}")
        lines.append(f"*Improvements:* {diff['improvement_count']}")
    else:
        lines.append("_First recorded run — no baseline comparison yet._")

    lines.append(f"*Avg summary score:* {stats['avg_summary_score']}/5")
    lines.append(f"*Avg latency:* {stats['avg_latency_ms']}ms")

    if report_url:
        lines.append(f"<{report_url}|View full diff report>")

    payload = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
        ]
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print("[slack_notifier] Alert sent successfully.")
        return True
    except requests.RequestException as e:
        print(f"[slack_notifier] Failed to send Slack alert: {e}")
        return False
