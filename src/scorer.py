"""
Multi-dimensional scoring: category match, LLM-judged summary quality,
latency, and token usage. Every dimension is stored per test case so the
comparator can diff on any of them later.
"""
from src.llm_client import judge_summary


async def score_result(test_case: dict, llm_result: dict) -> dict:
    """
    Combines the raw LLM output with the expected values from the golden
    dataset to produce a fully scored TestResult dict.
    """
    expected = test_case["expected"]
    email_text = test_case["input"]["email_text"]

    if llm_result["error"] or not llm_result["parsed"]:
        # The call itself failed — record as a hard failure, don't crash the run.
        return {
            "test_case_id": test_case["id"],
            "input_text": email_text,
            "expected_category": expected["category"],
            "actual_category": "ERROR",
            "category_match": False,
            "expected_summary": expected["summary"],
            "actual_summary": "",
            "summary_score": 1,
            "latency_ms": llm_result.get("latency_ms", 0),
            "prompt_tokens": llm_result.get("prompt_tokens", 0),
            "completion_tokens": llm_result.get("completion_tokens", 0),
            "difficulty": test_case.get("difficulty", "easy"),
            "error": llm_result["error"],
        }

    parsed = llm_result["parsed"]
    actual_category = parsed.get("category", "ERROR")
    actual_summary = parsed.get("summary", "")

    category_match = actual_category == expected["category"]

    judge_result = await judge_summary(expected["summary"], actual_summary)
    summary_score = judge_result.get("score", 1)

    return {
        "test_case_id": test_case["id"],
        "input_text": email_text,
        "expected_category": expected["category"],
        "actual_category": actual_category,
        "category_match": category_match,
        "expected_summary": expected["summary"],
        "actual_summary": actual_summary,
        "summary_score": summary_score,
        "latency_ms": round(llm_result["latency_ms"], 1),
        "prompt_tokens": llm_result["prompt_tokens"],
        "completion_tokens": llm_result["completion_tokens"],
        "difficulty": test_case.get("difficulty", "easy"),
        "error": None,
    }


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Rolls up per-case results into summary numbers for one run."""
    total = len(results)
    if total == 0:
        return {}

    category_matches = sum(1 for r in results if r["category_match"])
    avg_summary_score = sum(r["summary_score"] for r in results) / total
    avg_latency = sum(r["latency_ms"] for r in results) / total
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    errors = sum(1 for r in results if r["error"])

    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_difficulty[diff] = {
                "count": len(subset),
                "pass_rate": sum(1 for r in subset if r["category_match"]) / len(subset),
            }

    return {
        "total_cases": total,
        "category_pass_rate": category_matches / total,
        "avg_summary_score": round(avg_summary_score, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "error_count": errors,
        "by_difficulty": by_difficulty,
    }
