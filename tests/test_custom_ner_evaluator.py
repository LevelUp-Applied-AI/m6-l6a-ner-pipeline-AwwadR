import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from custom_ner_evaluator import (
    evaluate_micro,
    categorize_errors,
    spans_overlap,
)


def make_df(rows):
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def test_exact_match_scores_perfect():
    pred = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])
    gold = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])

    metrics = evaluate_micro(pred, gold, strategy="exact")

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_empty_predictions_gives_zero_recall():
    pred = make_df([])
    gold = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])

    metrics = evaluate_micro(pred, gold, strategy="exact")

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_negatives"] == 1


def test_empty_gold_standard_gives_zero_precision():
    pred = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])
    gold = make_df([])

    metrics = evaluate_micro(pred, gold, strategy="exact")

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_positives"] == 1


def test_partial_match_with_overlapping_span():
    pred = make_df([
        [1, "Paris", "LAW", 0, 5],
    ])
    gold = make_df([
        [1, "Paris Agreement", "LAW", 0, 15],
    ])

    metrics = evaluate_micro(pred, gold, strategy="partial")

    assert metrics["true_positives"] == 1
    assert metrics["f1"] == 1.0


def test_type_agnostic_match_ignores_label():
    pred = make_df([
        [1, "Jordan", "LOC", 0, 6],
    ])
    gold = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])

    metrics = evaluate_micro(pred, gold, strategy="type_agnostic")

    assert metrics["true_positives"] == 1
    assert metrics["f1"] == 1.0


def test_error_analysis_detects_type_error():
    pred = make_df([
        [1, "Jordan", "LOC", 0, 6],
    ])
    gold = make_df([
        [1, "Jordan", "GPE", 0, 6],
    ])

    errors = categorize_errors(pred, gold)

    assert "type_error" in errors["error_type"].values


def test_span_overlap_true_for_multi_token_overlap():
    pred = {
        "text_id": 1,
        "entity_text": "Paris",
        "entity_label": "LAW",
        "start_char": 0,
        "end_char": 5,
    }

    gold = {
        "text_id": 1,
        "entity_text": "Paris Agreement",
        "entity_label": "LAW",
        "start_char": 0,
        "end_char": 15,
    }

    assert spans_overlap(pred, gold) is True