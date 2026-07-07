"""
Test runner: loads the golden dataset, runs every case through the classifier
using async batching, and hands results to the scorer.
"""
import json
import asyncio
from pathlib import Path
from src.classifier import classify_email, load_prompt_config
from src.scorer import score_result

GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "golden_dataset" / "test_cases.json"

# Groq free tier has rate limits — cap concurrency to stay safely under them.
MAX_CONCURRENCY = 5


def load_golden_dataset() -> dict:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)


async def _run_single_case(test_case: dict, prompt_config: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        email_text = test_case["input"]["email_text"]
        llm_result = await classify_email(email_text, prompt_config)
        scored = await score_result(test_case, llm_result)
        return scored


async def run_eval(prompt_version: str) -> list[dict]:
    """
    Runs the full golden dataset through the classifier for the given
    prompt version (e.g. 'v1', 'v2'). Returns a list of TestResult dicts.
    """
    prompt_config = load_prompt_config(prompt_version)
    dataset = load_golden_dataset()
    test_cases = dataset["test_cases"]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [_run_single_case(tc, prompt_config, semaphore) for tc in test_cases]

    print(f"Running {len(tasks)} test cases against prompt version '{prompt_version}'...")
    results = await asyncio.gather(*tasks)
    print(f"Done. {len(results)} results collected.")

    return results


if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    results = asyncio.run(run_eval(version))
    for r in results:
        status = "PASS" if r["category_match"] else "FAIL"
        print(f"[{status}] {r['test_case_id']} — expected={r['expected_category']} actual={r['actual_category']} summary_score={r['summary_score']}")
