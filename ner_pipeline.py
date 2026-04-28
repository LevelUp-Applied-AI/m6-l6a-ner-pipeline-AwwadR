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
import math
import os 
import matplotlib.pyplot as plt
from transformers import pipeline as hf_pipeline
from itertools import combinations


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

ENTITY_NORMALIZATION_MAP = {
    "UN": "United Nations",
    "U.N.": "United Nations",
    "United Nations": "United Nations",
    "IPCC": "IPCC",
    "Intergovernmental Panel on Climate Change": "IPCC",
    "COP 28": "COP28",
    "COP28": "COP28",
    "COP27": "COP27",
    "Paris Agreement": "Paris Agreement",
    "Dead Sea": "Dead Sea",
}

def normalize_entities(entities_df, normalization_map):
    """Normalize entity variants into canonical names."""
    normalized_df = entities_df.copy()

    normalized_df["normalized_entity"] = (
        normalized_df["entity_text"]
        .str.strip()
        .replace(normalization_map)
    )

    return normalized_df

def compute_entity_cooccurrence(entities_df):
    """Compute how often pairs of normalized entities appear in the same text."""
    cooccurrence_counts = {}

    for text_id, group in entities_df.groupby("text_id"):
        unique_entities = sorted(set(group["normalized_entity"]))

        for entity_a, entity_b in combinations(unique_entities, 2):
            pair = (entity_a, entity_b)
            cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1

    edge_list = []

    for (entity_a, entity_b), count in cooccurrence_counts.items():
        edge_list.append({
            "entity_1": entity_a,
            "entity_2": entity_b,
            "cooccurrence_count": count
        })

    return pd.DataFrame(edge_list).sort_values(
        by="cooccurrence_count",
        ascending=False
    )

def add_category_to_entities(entities_df, articles_df):
    """Add article category to each extracted entity using text_id."""
    category_map = articles_df[["id", "category"]].rename(columns={"id": "text_id"})

    return entities_df.merge(category_map, on="text_id", how="left")

def compute_entity_importance_by_category(entities_df):
    """Compute TF-IDF-style importance of entities within each category."""
    category_entity_counts = (
        entities_df
        .groupby(["category", "normalized_entity"])
        .size()
        .reset_index(name="term_frequency")
    )

    total_categories = entities_df["category"].nunique()

    entity_category_counts = (
        entities_df[["category", "normalized_entity"]]
        .drop_duplicates()
        .groupby("normalized_entity")
        .size()
        .reset_index(name="category_document_frequency")
    )

    importance_df = category_entity_counts.merge(
        entity_category_counts,
        on="normalized_entity",
        how="left"
    )

    importance_df["idf"] = importance_df["category_document_frequency"].apply(
        lambda df_count: math.log((total_categories + 1) / (df_count + 1)) + 1
    )

    importance_df["importance_score"] = (
        importance_df["term_frequency"] * importance_df["idf"]
    )

    return importance_df.sort_values(
        by="importance_score",
        ascending=False
    )

def plot_entity_network(edge_df, output_path="outputs/tier2_entity_network.png", top_n=20):
    """Plot a simple network visualization of top entity co-occurrences."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_edges = edge_df.head(top_n)

    entities = sorted(set(top_edges["entity_1"]) | set(top_edges["entity_2"]))

    angle_step = 2 * math.pi / len(entities)
    positions = {}

    for i, entity in enumerate(entities):
        angle = i * angle_step
        positions[entity] = (math.cos(angle), math.sin(angle))

    plt.figure(figsize=(12, 12))

    for _, row in top_edges.iterrows():
        entity_1 = row["entity_1"]
        entity_2 = row["entity_2"]
        count = row["cooccurrence_count"]

        x_values = [positions[entity_1][0], positions[entity_2][0]]
        y_values = [positions[entity_1][1], positions[entity_2][1]]

        plt.plot(x_values, y_values, linewidth=max(1, count / 2), alpha=0.6)

    for entity, (x, y) in positions.items():
        plt.scatter(x, y, s=300)
        plt.text(x, y, entity, fontsize=9, ha="center", va="center")

    plt.title("Top 20 Entity Co-occurrences")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

def save_tier2_outputs(edge_df, importance_df):
    """Save Tier 2 aggregation outputs."""
    os.makedirs("outputs", exist_ok=True)

    edge_df.to_csv("outputs/tier2_entity_cooccurrence_edges.csv", index=False)
    importance_df.to_csv("outputs/tier2_entity_importance.csv", index=False)


if __name__ == "__main__":
    # Load spaCy once and reuse across functions
    nlp = spacy.load("en_core_web_sm")

    # Try to load Hugging Face NER.
    # If it fails because of internet/cache issues, continue with spaCy + Tier 2.
    try:
        hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")
    except Exception as e:
        print(f"Could not load Hugging Face NER: {e}")
        hf_ner = None

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

        # HF NER across the English corpus only if HF loaded successfully
        if hf_ner is not None:
            hf_entities = extract_hf_entities(df, hf_ner)

            if hf_entities is not None:
                print(f"HF entities: {len(hf_entities)} total")
        else:
            hf_entities = None

        # Compare the two systems only if both outputs exist
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
            hf_metrics = evaluate_ner(hf_entities, gold)

            if hf_metrics is not None:
                print(f"hf evaluation: {hf_metrics}")
        else:
            print("Skipping HF evaluation because Hugging Face NER did not load.")

        # Tier 2: Custom Entity Aggregation Pipeline
        if spacy_entities is not None:
            print("\nTier 2: Custom Entity Aggregation Pipeline")

            spacy_entities_with_category = add_category_to_entities(spacy_entities, df)

            normalized_entities = normalize_entities(
                spacy_entities_with_category,
                ENTITY_NORMALIZATION_MAP
            )

            print("\nSample normalized entities:")
            print(
                normalized_entities[
                    ["entity_text", "normalized_entity", "category"]
                ].head(10)
            )

            cooccurrence_edges = compute_entity_cooccurrence(normalized_entities)

            print("\nTop entity co-occurrences:")
            print(cooccurrence_edges.head(20))

            entity_importance = compute_entity_importance_by_category(
                normalized_entities
            )

            print("\nTop entity importance scores:")
            print(entity_importance.head(20))

            save_tier2_outputs(cooccurrence_edges, entity_importance)

            network_path = plot_entity_network(cooccurrence_edges)

            print(
                "\nTier 2 co-occurrence edges saved to: "
                "outputs/tier2_entity_cooccurrence_edges.csv"
            )
            print(
                "Tier 2 entity importance saved to: "
                "outputs/tier2_entity_importance.csv"
            )
            print(f"Tier 2 network visualization saved to: {network_path}")