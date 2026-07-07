# Model Regression Detection System

CI/CD for prompt changes. This service runs any LLM-powered feature against a
hand-labeled golden dataset whenever a prompt changes, diffs the results
against the last known-good run, and alerts the team in Slack before a bad
prompt reaches production.

If you're new to this repo, read this doc top to bottom before touching
anything — it'll save you time.

## What problem this solves

Prompt changes usually ship blind. Someone tweaks a system prompt, it looks
fine on the three examples they tried by hand, and it merges. Two weeks
later someone notices the classifier has been quietly misrouting 8% of
account-deletion emails as "general." This system exists so that gap never
opens in the first place — every prompt change gets run against a fixed set
of known-correct examples before it merges, and any case that used to pass
and now fails gets flagged immediately, with a diff, in Slack.

## Architecture, in one paragraph

`main.py` loads a versioned prompt from `/prompts`, runs it against every
case in `/golden_dataset/test_cases.json` through Groq's API
(`src/runner.py` → `src/classifier.py` → `src/llm_client.py`), scores each
result on category accuracy and LLM-judged summary quality
(`src/scorer.py`), diffs it against the previous run stored in SQLite
(`src/comparator.py`, `src/db.py`), checks for slow multi-run drift
(`src/drift_tracker.py`), renders an HTML report
(`reports/report_generator.py`), and posts a summary to Slack
(`alerts/slack_notifier.py`). GitHub Actions triggers this on every PR that
touches `/prompts` or `/golden_dataset`, and blocks merge if the diff is
classified `critical`.

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in:
   - `GROQ_API_KEY` — free, no credit card, get one at
     https://console.groq.com/keys
   - `SLACK_WEBHOOK_URL` — optional, leave blank to skip Slack alerts.
     Create one at https://api.slack.com/messaging/webhooks

3. Run your first eval:
   ```
   python main.py --prompt-version v1
   ```

4. Open the report:
   ```
   open reports/output/latest.html
   ```

5. (Optional) Run the dashboard:
   ```
   streamlit run dashboard/app.py
   ```

## Running a regression on purpose (to see the system work)

`prompts/v2_classifier.yaml` is a deliberately weaker prompt — shorter,
fewer category definitions, no few-shot examples. Run it after `v1` to see
the diffing actually catch something:

```
python main.py --prompt-version v1   # establishes baseline
python main.py --prompt-version v2   # compare against v1, expect regressions
```

## How to add a new test case to the golden dataset

Open `golden_dataset/test_cases.json` and add an entry:

```json
{
  "id": "tc_016",
  "input": {"email_text": "..."},
  "expected": {"category": "billing", "summary": "..."},
  "difficulty": "medium",
  "notes": "Why this case exists / what it's testing for."
}
```

Rules that keep this dataset useful:

- **Never generate expected labels with an LLM.** The whole point of a
  golden dataset is that it's human-verified ground truth. If the labels
  come from a model, you're just testing the model against itself.
- Every new case needs a `notes` field explaining why it's there. Six
  months from now nobody will remember why `tc_016` exists unless you write
  it down now.
- Prefer adding edge cases over easy ones. The dataset is more valuable the
  more it stresses ambiguity, typos, sarcasm, short input, and mixed
  intent — that's where prompts actually break.
- When you find a real failure in production, add it as a test case. This
  is how the dataset should grow over time — seeded by hand, expanded by
  failure cases.

## How to adjust thresholds

Edit `config/thresholds.yaml`:

```yaml
warning_threshold_pct: 3    # pass-rate drop that triggers a warning
critical_threshold_pct: 8   # pass-rate drop that blocks merge
drift_window_runs: 7        # rolling window size for drift detection
drift_threshold_pct: 5      # rolling-average drop that triggers a drift alert
```

Thresholds are percentage points, not percentages of the pass rate. A drop
from 94% to 91% is a 3-point delta, not a 3% relative change.

## Why we track drift separately from per-run regressions

A single prompt change can cause an obvious, immediate regression — that's
easy to catch with a run-over-run diff. But a slower failure mode also
exists: five small "harmless" tweaks across a month, each moving the pass
rate down by half a point, none individually crossing the warning
threshold. By the time anyone notices, the classifier has drifted 5+ points
without ever tripping a single-run alert. `drift_tracker.py` guards against
this by comparing a rolling 7-run average against the historical average
before that window, independent of what any single run looked like.

## Why the LLM-as-judge uses a smaller model than the feature under test

`src/llm_client.py` uses `llama-3.3-70b-versatile` for the classifier being
tested, but `llama-3.1-8b-instant` for judging summary quality. Judging
"does this summary capture the same idea as the reference summary" is a
much easier task than the classification itself, and running 15-100 judge
calls per eval run adds up fast even on a free tier. Using the cheaper,
faster model here keeps the full suite fast without meaningfully hurting
judge reliability — that tradeoff won't hold for every use case, so
reconsider it if you're evaluating something more nuanced than short-text
similarity.

## Running in CI

`.github/workflows/eval-on-pr.yml` triggers automatically on any PR that
touches `/prompts` or `/golden_dataset`. It needs two repo secrets set
under Settings → Secrets → Actions:

- `GROQ_API_KEY`
- `SLACK_WEBHOOK_URL` (optional)

If the run comes back `critical`, `main.py` exits with a non-zero status
code, which fails the GitHub Action and blocks merge.

## Running with Docker

```
docker build -t model-regression-detector .
docker run --env-file .env model-regression-detector python main.py --prompt-version v1
```

Or with docker-compose (runs the eval and the dashboard together):

```
docker-compose up
```

## What's deliberately out of scope

This is a reference implementation, not a general-purpose eval platform.
It's scoped to one feature (email classification) with one dataset format
and one LLM provider. If you're extending this to a second feature, the
right move is probably to parameterize `golden_dataset/` and `prompts/` per
feature rather than trying to generalize the schema — keep the golden
dataset format simple and feature-specific; that's what makes it easy to
hand-label in the first place.
