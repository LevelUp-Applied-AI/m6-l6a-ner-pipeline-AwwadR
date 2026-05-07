"""
Module 6 Week B — Stretch: Cross-Lingual Embedding Comparison

This script compares English and Arabic climate texts using multilingual BERT.
It selects English and Arabic texts from climate_articles.csv, computes embeddings,
creates a cosine similarity matrix, saves a heatmap, and prints useful similarity
scores for analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/climate_articles.csv"
OUTPUT_DIR = "outputs"
HEATMAP_PATH = os.path.join(OUTPUT_DIR, "cross_lingual_similarity_heatmap.png")
MODEL_NAME = "bert-base-multilingual-cased"


def load_bilingual_data(filepath):
    """Load climate articles and return English and Arabic subsets."""
    df = pd.read_csv(filepath)

    df = df.dropna(subset=["text", "language"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    english_df = df[df["language"] == "en"].copy().reset_index(drop=True)
    arabic_df = df[df["language"] == "ar"].copy().reset_index(drop=True)

    return english_df, arabic_df


def select_texts(english_df, arabic_df, n=10):
    """Select at least 10 English and 10 Arabic texts.

    This simple version selects the first n texts from each language.
    You can improve this later by manually choosing same-topic pairs.
    """

    english_texts = english_df["text"].head(n).tolist()
    arabic_texts = arabic_df["text"].head(n).tolist()

    return english_texts, arabic_texts


def mean_pool_embeddings(texts, tokenizer, model):
    """Compute mean-pooled multilingual BERT embeddings."""
    embeddings = []

    model.eval()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )

            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            summed_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)

            mean_embedding = summed_embeddings / token_counts

            embeddings.append(mean_embedding.squeeze().cpu().numpy())

    return np.array(embeddings)


def make_labels(texts, max_chars=40):
    """Create short labels for heatmap axes."""
    labels = []

    for i, text in enumerate(texts, start=1):
        clean_text = text.replace("\n", " ").strip()
        labels.append(f"{i}. {clean_text[:max_chars]}")
    
    return labels


def plot_similarity_heatmap(similarity_matrix, labels, output_path):
    """Save a 20x20 cosine similarity heatmap."""
    plt.figure(figsize=(16, 14))

    plt.imshow(similarity_matrix, aspect="auto")
    plt.colorbar(label="Cosine Similarity")

    plt.xticks(
        ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=7
    )

    plt.yticks(
        ticks=np.arange(len(labels)), labels=labels, fontsize=7
    )

    plt.title("Cross-Lingual Similarity Heatmap: English and Arabic Climate Texts")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_top_cross_lingual_pairs(similarity_matrix, english_texts, arabic_texts, top_k=10):
    """Print top English-Arabic similarity pairs.

    English texts are first 10 rows/columns.
    Arabic texts are next 10 rows/columns.
    """
    n_en = len(english_texts)

    cross_scores = []

    for en_idx in range(n_en):
        for ar_idx in range(len(arabic_texts)):
            matrix_ar_idx = n_en + ar_idx
            score = similarity_matrix[en_idx, matrix_ar_idx]

            cross_scores.append({
                "english_index": en_idx,
                "arabic_index": ar_idx,
                "score": float(score),
                "english_preview": english_texts[en_idx][:100].replace("\n", " "),
                "arabic_preview": arabic_texts[ar_idx][:100].replace("\n", " ")
            })
    
    cross_scores = sorted(cross_scores, key=lambda x: x["score"], reverse=True)

    print("\nTop English-Arabic Similarity Pairs")
    print("=" * 80)

    for i, item in enumerate(cross_scores[:top_k], start=1):
        print(f"\nPair {i}")
        print(f"Score: {item['score']:.4f}")
        print(f"English {item['english_index'] + 1}: {item['english_preview']}...")
        print(f"Arabic {item['arabic_index'] + 1}: {item['arabic_preview']}...")
    
    return cross_scores

def compare_within_and_cross_language(similarity_matrix, n_en, n_ar):
    """Compare average English-English, Arabic-Arabic, and English-Arabic similarity."""
    english_block = similarity_matrix[:n_en, :n_en]
    arabic_block = similarity_matrix[n_en:n_en + n_ar, n_en:n_en + n_ar]
    cross_block = similarity_matrix[:n_en, n_en:n_en + n_ar]

    # Remove diagonal for within-language averages
    english_scores = english_block[np.triu_indices(n_en, k=1)]
    arabic_scores = arabic_block[np.triu_indices(n_ar, k=1)]
    cross_scores = cross_block.flatten()

    summary = {
        "english_english_mean": float(np.mean(english_scores)),
        "arabic_arabic_mean": float(np.mean(arabic_scores)),
        "english_arabic_mean": float(np.mean(cross_scores)),
        "english_english_max": float(np.max(english_scores)),
        "arabic_arabic_max": float(np.max(arabic_scores)),
        "english_arabic_max": float(np.max(cross_scores)),
        "english_arabic_min": float(np.min(cross_scores)),
    }

    print("\nSimilarity Summary")
    print("=" * 80)
    print(f"English-English mean similarity: {summary['english_english_mean']:.4f}")
    print(f"Arabic-Arabic mean similarity:   {summary['arabic_arabic_mean']:.4f}")
    print(f"English-Arabic mean similarity:  {summary['english_arabic_mean']:.4f}")
    print(f"English-English max similarity:  {summary['english_english_max']:.4f}")
    print(f"Arabic-Arabic max similarity:    {summary['arabic_arabic_max']:.4f}")
    print(f"English-Arabic max similarity:   {summary['english_arabic_max']:.4f}")
    print(f"English-Arabic min similarity:   {summary['english_arabic_min']:.4f}")

    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading bilingual climate dataset...")
    english_df, arabic_df = load_bilingual_data(DATA_PATH)

    print(f"English texts available: {len(english_df)}")
    print(f"Arabic texts available: {len(arabic_df)}")

    english_texts, arabic_texts = select_texts(english_df, arabic_df, n=10)

    print(f"Selected English texts: {len(english_texts)}")
    print(f"Selected Arabic texts: {len(arabic_texts)}")

    all_texts = english_texts + arabic_texts

    print("\nLoading multilingual BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("\nComputing multilingual embeddings...")
    embeddings = mean_pool_embeddings(all_texts, tokenizer, model)
    print(f"Embedding matrix shape: {embeddings.shape}")

    print("\nComputing 20x20 cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    labels = make_labels(all_texts)

    print("\nSaving heatmap...")
    plot_similarity_heatmap(similarity_matrix, labels, HEATMAP_PATH)
    print(f"Saved heatmap to {HEATMAP_PATH}")

    cross_scores = print_top_cross_lingual_pairs(
        similarity_matrix,
        english_texts,
        arabic_texts,
        top_k=10
    )

    summary = compare_within_and_cross_language(
        similarity_matrix,
        n_en=len(english_texts),
        n_ar=len(arabic_texts)
    )

    print("\nDone.")
    print("Use the printed scores and the heatmap in stretch_analysis.md.")


if __name__ == "__main__":
    main()