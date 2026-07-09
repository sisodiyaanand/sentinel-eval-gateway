"""
Unit tests for evaluation metrics.
Verifies that similarity scores behave correctly across
identical, related, and unrelated text pairs.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evals.metrics import (
    compute_semantic_similarity,
    compute_toxicity_score,
    compute_hallucination_proxy,
)


def test_identical_text_has_max_similarity():
    score = compute_semantic_similarity(
        "Explain how firewalls work",
        "Explain how firewalls work",
    )
    assert score == 1.0


def test_related_text_has_partial_similarity():
    score = compute_semantic_similarity(
        "Explain how firewalls work",
        "A firewall filters network traffic based on security rules.",
    )
    assert 0.0 < score < 0.5


def test_unrelated_text_has_near_zero_similarity():
    score = compute_semantic_similarity(
        "Explain how firewalls work",
        "I like eating pizza on weekends",
    )
    assert score == 0.0


def test_empty_input_returns_zero():
    assert compute_semantic_similarity("", "something") == 0.0
    assert compute_semantic_similarity("something", "") == 0.0


def test_toxicity_score_detects_toxic_keywords():
    score = compute_toxicity_score("You are so stupid and worthless")
    assert score > 0.0


def test_toxicity_score_clean_text_is_zero():
    score = compute_toxicity_score("Firewalls filter network traffic")
    assert score == 0.0


def test_hallucination_proxy_without_context_returns_zero():
    score = compute_hallucination_proxy("Some response")
    assert score == 0.0


def test_hallucination_proxy_with_matching_context_is_low():
    score = compute_hallucination_proxy(
        "Firewalls filter network traffic",
        context="Firewalls filter network traffic based on rules",
    )
    assert score < 0.7