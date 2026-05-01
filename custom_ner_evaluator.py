import os
import pandas as pd

def dataframe_to_records(df):
    """Convert entity DataFrame into a list of clean entity dictionaries."""
    required_columns = ["text_id", "entity_text", "entity_label", "start_char", "end_char"]

    if df is None or df.empty:
        return []
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    records = []

    for _, row in df.iterrows():
        records.append({
            "text_id": row["text_id"],
            "entity_text": row["entity_text"],
            "entity_label":row["entity_label"],
            "start_char": int(row["start_char"]),
            "end_char": int(row["end_char"])
        })
    
    return records

def spans_overlap(pred, gold):
    """Return True if predicted and gold spans overlap in the same text."""
    if pred["text_id"] != gold["text_id"]:
        return False
    
    return pred["start_char"] < gold["end_char"] and gold["start_char"] < pred["end_char"]

def is_exact_match(pred, gold):
    """Check exact span and label match."""
    return (
        pred["text_id"] == gold["text_id"]
        and pred["start_char"] == gold["start_char"]
        and pred["end_char"] == gold["end_char"]
        and pred["entity_label"] == gold["entity_label"]
    )

def is_type_agnostic_match(pred, gold):
    """Check exact match while ignoring entity label."""
    return (
        pred["text_id"] == gold["text_id"]
        and pred["start_char"] == gold["start_char"]
        and pred["end_char"] == gold["end_char"]
    )

def is_partial_match(pred, gold):
    """Check overlapping span with same label."""
    return (
        spans_overlap(pred, gold)
        and pred["entity_label"] == gold["entity_label"]
    )

def match_entities(pred_records, gold_records, strategy="exact"):
    """Match predicted entities to gold entities using a selected strategy."""
    matched_gold_indexes = set()
    true_positives = 0

    for pred in pred_records:
        best_match_index = None

        for gold_index, gold in enumerate(gold_records):
            if gold_index in matched_gold_indexes:
                continue
            if strategy == "exact":
                matched = is_exact_match(pred, gold)
            elif strategy == "partial":
                matched = is_partial_match(pred, gold)
            elif strategy == "type_agnostic":
                matched = is_type_agnostic_match(pred, gold)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if matched:
                best_match_index = gold_index
                break
        
        if best_match_index is not None:
            true_positives += 1
            matched_gold_indexes.add(best_match_index)
    
    false_positives = len(pred_records) - true_positives
    false_negatives = len(gold_records) - true_positives

    return true_positives, false_positives, false_negatives

def compute_metrics(true_positives, false_positives, false_negatives):
    """Compute precision, recall, and F1 safely."""
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def evaluate_micro(predicted_df, gold_df, strategy="exact"):
    """Evaluate all entities together using micro averaging."""
    pred_records = dataframe_to_records(predicted_df)
    gold_records = dataframe_to_records(gold_df)

    tp, fp, fn = match_entities(pred_records, gold_records, strategy=strategy)

    return compute_metrics(tp, fp, fn)

def evaluate_macro(predicted_df, gold_df, strategy="exact"):
    """Evaluate per text_id, then average precision, recall, and f1."""
    pred_records = dataframe_to_records(predicted_df)
    gold_records = dataframe_to_records(gold_df)

    text_ids = sorted(
        set([record["text_id"] for record in pred_records]) | set(record["text_id"] for record in gold_records)
    )

    if not text_ids:
        return{
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "num_texts":0
        }
    
    per_text_metrics = []

    for text_id in text_ids:
        pred_text = [record for record in pred_records if record["text_id"] == text_id]
        gold_text = [record for record in gold_records if record["text_id"] == text_id]

        tp, fp, fn = match_entities(pred_text, gold_text, strategy=strategy)
        metrics = compute_metrics(tp, fp, fn)

        per_text_metrics.append(metrics)

    return {
        "precision": sum(m["precision"] for m in per_text_metrics) / len(per_text_metrics),
        "recall": sum(m["recall"] for m in per_text_metrics) / len(per_text_metrics),
        "f1": sum(m["f1"] for m in per_text_metrics) / len(per_text_metrics),
        "num_texts": len(text_ids)
    }

def evaluate_all_strategies(predicted_df, gold_df):
    """Evaluate exact, partial, and type-agnostic matching with micro and macro averaging."""
    strategies = ["exact", "partial", "type_agnostic"]
    rows = []

    for strategy in strategies:
        micro_metrics = evaluate_micro(predicted_df, gold_df, strategy=strategy)
        macro_metrics = evaluate_macro(predicted_df, gold_df, strategy=strategy)

        rows.append({
            "strategy": strategy, 
            "averaging": "micro", 
            "precision": micro_metrics["precision"],
            "recall": micro_metrics["recall"],
            "f1": micro_metrics["f1"],
            "true_positives": micro_metrics.get("true_positives"),
            "false_positives": micro_metrics.get("false_positives"),
            "false_negatives": micro_metrics.get("false_negatives")
        })

        rows.append({
            "strategy": strategy,
            "averaging": "macro",
            "precision": macro_metrics["precision"],
            "recall": macro_metrics["recall"],
            "f1": macro_metrics["f1"],
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
        })
    
    return pd.DataFrame(rows)

def categorize_errors(predicted_df, gold_df):
    """Categorize NER errors into boundary, type, missing, and spurious errors."""
    pred_records = dataframe_to_records(predicted_df)
    gold_records = dataframe_to_records(gold_df)

    errors = []

    matched_pred_indexes = set()
    matched_gold_indexes = set()

    # First remove exact matches because they are correct, not errors
    for pred_index, pred in enumerate(pred_records):
        for gold_index, gold in enumerate(gold_records):
            if gold_index in matched_gold_indexes:
                continue
            if is_exact_match(pred, gold):
                matched_pred_indexes.add(pred_index)
                matched_gold_indexes.add(gold_index)
                break
    
    # Analyze remaining predictions
    for pred_index, pred in enumerate(pred_records):
        if pred_index in matched_pred_indexes:
            continue

        same_span_wrong_label = None
        overlapping_same_label = None

        for gold_index, gold in enumerate(gold_records):
            if gold_index in matched_gold_indexes:
                continue

            if is_type_agnostic_match(pred, gold) and pred["entity_label"] != gold["entity_label"]:
                same_span_wrong_label = gold
                break

            if spans_overlap(pred, gold) and pred["entity_label"] == gold["entity_label"]:
                overlapping_same_label = gold
                break
        
        if same_span_wrong_label is not None:
            errors.append({
                "error_type": "type_error",
                "text_id": pred["text_id"],
                "predicted_text": pred["entity_text"],
                "predicted_label": pred["entity_label"],
                "gold_text": same_span_wrong_label["entity_text"],
                "gold_label": same_span_wrong_label["entity_label"],
            })
            matched_pred_indexes.add(pred_index)
        
        elif overlapping_same_label is not None:
            errors.append({
                "error_type": "boundary_error",
                "text_id": pred["text_id"],
                "predicted_text": pred["entity_text"],
                "predicted_label": pred["entity_label"],
                "gold_text": overlapping_same_label["entity_text"],
                "gold_label": overlapping_same_label["entity_label"],
            })
            matched_pred_indexes.add(pred_index)
        
        else:
            errors.append({
                "error_type": "spurious_entity",
                "text_id": pred["text_id"],
                "predicted_text": pred["entity_text"],
                "predicted_label": pred["entity_label"],
                "gold_text": None,
                "gold_label": None,
            })
            matched_pred_indexes.add(pred_index)
    
    # Gold entities that were never matched are missing
    for gold_index, gold in enumerate(gold_records):
        if gold_index not in matched_gold_indexes:
            errors.append({
                "error_type": "missing_entity",
                "text_id": gold["text_id"],
                "predicted_text": None,
                "predicted_label": None,
                "gold_text": gold["entity_text"],
                "gold_label": gold["entity_label"],
            })
    
    return pd.DataFrame(errors)

def create_error_distribution_report(errors_df):
    """Count errors by error type."""
    if errors_df.empty:
        return pd.DataFrame(columns=["error_type", "count"])
    
    return (
        errors_df["error_type"].value_counts().reset_index().rename(columns={"index": "error_type", "count": "count"})
    )

if __name__ == "__main__":
    from ner_pipeline import load_data, extract_spacy_entities
    import spacy

    os.makedirs("outputs", exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    df = load_data("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")

    spacy_entities = extract_spacy_entities(df, nlp)

    evaluation_results = evaluate_all_strategies(spacy_entities, gold_df)
    errors = categorize_errors(spacy_entities, gold_df)
    error_report = create_error_distribution_report(errors)

    evaluation_results.to_csv("outputs/tier3_custom_evaluator_metrics.csv", index=False)
    errors.to_csv("outputs/tier3_error_analysis.csv", index=False)
    error_report.to_csv("outputs/tier3_error_distribution.csv", index=False)

    print("\nTier 3 evaluation results:")
    print(evaluation_results)

    print("\nTier 3 error distribution:")
    print(error_report)

    print("\nSaved Tier 3 outputs:")
    print("- outputs/tier3_custom_evaluator_metrics.csv")
    print("- outputs/tier3_error_analysis.csv")
    print("- outputs/tier3_error_distribution.csv")
