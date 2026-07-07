"""
Basic sanity tests for the eval pipeline itself — not the LLM's outputs.
Run with: pytest tests/
"""
import pytest
from src.scorer import compute_aggregate_stats
from src.comparator import compare_runs


def _fake_result(tc_id, category_match, expected_category="billing", difficulty="easy", score=5):
    return {
        "test_case_id": tc_id,
        "input_text": "some email",
        "expected_category": expected_category,
        "actual_category": expected_category if category_match else "technical",
        "category_match": category_match,
        "expected_summary": "expected",
        "actual_summary": "actual",
        "summary_score": score,
        "latency_ms": 100.0,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "difficulty": difficulty,
        "error": None,
    }


def test_compute_aggregate_stats_basic():
    results = [
        _fake_result("tc1", True),
        _fake_result("tc2", True),
        _fake_result("tc3", False),
    ]
    stats = compute_aggregate_stats(results)
    assert stats["total_cases"] == 3
    assert stats["category_pass_rate"] == pytest.approx(2 / 3)
    assert stats["avg_summary_score"] == 5.0


def test_compute_aggregate_stats_empty():
    assert compute_aggregate_stats([]) == {}


def test_compare_runs_no_baseline():
    results = [_fake_result("tc1", True)]
    stats = compute_aggregate_stats(results)
    diff = compare_runs(results, stats, None, None)
    assert diff["has_baseline"] is False
    assert diff["severity"] == "info"


def test_compare_runs_detects_regression():
    baseline_results = [_fake_result("tc1", True), _fake_result("tc2", True)]
    baseline_stats = compute_aggregate_stats(baseline_results)

    current_results = [_fake_result("tc1", True), _fake_result("tc2", False)]
    current_stats = compute_aggregate_stats(current_results)

    diff = compare_runs(current_results, current_stats, baseline_results, baseline_stats)

    assert diff["has_baseline"] is True
    assert diff["regression_count"] == 1
    assert diff["regressions"][0]["test_case_id"] == "tc2"
    assert diff["overall_pass_rate_delta"] < 0


def test_compare_runs_detects_improvement():
    baseline_results = [_fake_result("tc1", False), _fake_result("tc2", True)]
    baseline_stats = compute_aggregate_stats(baseline_results)

    current_results = [_fake_result("tc1", True), _fake_result("tc2", True)]
    current_stats = compute_aggregate_stats(current_results)

    diff = compare_runs(current_results, current_stats, baseline_results, baseline_stats)

    assert diff["improvement_count"] == 1
    assert diff["improvements"][0]["test_case_id"] == "tc1"
