"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
import unicodedata
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df["text"].str.split().str.len()

    return {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": word_counts.mean(),
            "min": word_counts.min(),
            "max": word_counts.max()
        }
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    normalized_text = unicodedata.normalize("NFC", text)
    doc = nlp(normalized_text)

    clean_tokens = []

    for token in doc:
        if not token.is_punct and not token.is_space:
            clean_tokens.append(token.lemma_.lower())
    
    return clean_tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df["language"] == "en"]
    entities = []

    for _, row in english_df.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            entities.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(
        entities,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"]
    )


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df["language"] == "en"]
    entities = []

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = row["text"]
        raw_entities = ner_pipeline(text)

        current_entity = None

        for item in raw_entities:
            word = item["word"]
            raw_label = item["entity"]
            clean_label = raw_label.replace("B-", "").replace("I-", "")

            if raw_label.startswith("B-") or current_entity is None:
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    "text_id": text_id,
                    "entity_text": word.replace("##", ""),
                    "entity_label": clean_label,
                    "start_char": item["start"],
                    "end_char": item["end"]
                }
            
            elif raw_label.startswith("I-") and current_entity is not None:
                if word.startswith("##"):
                    current_entity["entity_text"] += word.replace("##", "")
                else:
                    current_entity["entity_text"] += " " + word
                current_entity["end_char"] = item["end"]
        
        if current_entity is not None:
            entities.append(current_entity)
    
    return pd.DataFrame(entities, columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"])


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    total_spacy = len(spacy_df)
    total_hf = len(hf_df)

    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    both = spacy_set & hf_set
    spacy_only = spacy_set - hf_set
    hf_only =  hf_set - spacy_set

    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": total_spacy,
        "total_hf": total_hf,
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    predicted_set = set(zip(
        predicted_df["text_id"],
        predicted_df["entity_text"],
        predicted_df["entity_label"]
    ))

    gold_set = set(zip(
        gold_df["text_id"],
        gold_df["entity_text"],
        gold_df["entity_label"]
    ))

    true_positive = len(predicted_set & gold_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
        if hf_entities is not None:
            metrics = evaluate_ner(hf_entities, gold)
            if metrics is not None:
                print(f"hf evaluation: {metrics}")