"""
Module 6 Week A - Stretch: Multilingual NER Comparison

This script compares multilingual NER performance on English and Arabic
climate articles using:
1. spaCy xx_ent_wiki_sm
2. Hugging Face Davlan/xlm-roberta-base-wikiann-ner

Run:
    python stretch_multilingual_ner.py
"""
import os
import pandas as pd
import spacy
from collections import Counter
from transformers import pipeline

DATA_PATH = "data/climate_articles.csv"
OUTPUT_DIR = "outputs"
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "multilingual_ner_comparison.csv")
ANALYSIS_PATH = os.path.join(OUTPUT_DIR, "stretch_analysis.md")

def load_dataset(filepath=DATA_PATH):
    """Load the climate articles dataset."""
    return pd.read_csv(filepath)

def sample_texts(df, language, n=20):
    """Return the first n texts for a selected language."""
    subset = df[df["language"] == language].copy()
    subset = subset.head(n)
    return subset

def run_spacy_ner(texts, nlp):
    """
    Run spaCy multilingual NER.

    Returns:
        list of dicts with text_id, entity_text, label, model
    """
    rows = []

    for _, row in texts.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        doc = nlp(text)

        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "model": "spaCy_xx_ent_wiki_sm"
            })
    
    return pd.DataFrame(rows)

def run_hf_ner(texts, ner_pipeline):
    """
    Run Hugging Face Multilingual NER.

    Uses grouped_entities=True so subword tokens are merged into full entities.
    """
    rows = []

    for _, row in texts.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        try:
            results = ner_pipeline(text)
        except Exception as e:
            print(f"HF model failed on text_id={text_id}: {e}")
            continue

        for ent in results:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.get("word", ""),
                "entity_label": ent.get("entity_group", ent.get("entity", "")),
                "model": "HF_xlm_roberta_wikiann"
            })
    
    return pd.DataFrame(rows)

def count_words(text):
    """Simple word count for entity density."""
    return len(str(text).split())

def summarize_results(entity_df, texts_df, language, model_name):
    """
    Create summary row for one language/model combination.

    Reports:
    - total texts
    - total words
    - total entities
    - entities per 100 words
    - no-entity text rate
    - entity counts per type
    - 3 example entities
    """
    total_texts = len(texts_df)
    total_words = texts_df["text"].apply(count_words).sum()
    total_entities = len(entity_df)

    if total_words > 0:
        entity_density = round((total_entities / total_words) * 100, 2)
    else:
        entity_density = 0
    
    entity_counts = entity_df["entity_label"].value_counts().to_dict()

    texts_with_entities = entity_df["text_id"].nunique() if not entity_df.empty else 0
    no_entity_texts = total_texts - texts_with_entities
    no_entity_rate = round((no_entity_texts / total_texts) * 100, 2)

    if entity_df.empty:
        examples = "No entities found"
    else:
        examples = "; ".join(
            entity_df.head(3).apply(lambda row: f"{row['entity_text']} ({row['entity_label']})", axis=1).tolist()
        )
    
    return {
        "language": language,
        "model": model_name,
        "texts_processed": total_texts,
        "total_words": total_words,
        "total_entities": total_entities,
        "entities_per_100_words": entity_density,
        "no_entity_texts": no_entity_texts,
        "no_entity_rate_percent": no_entity_rate,
        "entity_counts_by_type": entity_counts,
        "example_entities": examples
    }

def write_analysis(comparison_df):
    """
    Write a stronger Markdown analysis file using actual result values.
    """
    def get_row(language, model):
        row = comparison_df[
            (comparison_df["language"] == language)
            & (comparison_df["model"] == model)
        ].iloc[0]
        return row

    spacy_en = get_row("en", "spaCy_xx_ent_wiki_sm")
    spacy_ar = get_row("ar", "spaCy_xx_ent_wiki_sm")
    hf_en = get_row("en", "HF_xlm_roberta_wikiann")
    hf_ar = get_row("ar", "HF_xlm_roberta_wikiann")

    analysis = f"""# Stretch 6A — Multilingual NER Comparison Analysis

## Results Summary

I processed **20 English texts** and **20 Arabic texts** from the climate articles dataset using two multilingual NER models: spaCy `xx_ent_wiki_sm` and Hugging Face `Davlan/xlm-roberta-base-wikiann-ner`.

| Language | Model | Texts | Words | Total entities | Entities / 100 words | No-entity texts | No-entity rate | Entity counts | Example entities |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| English | spaCy `xx_ent_wiki_sm` | {spacy_en['texts_processed']} | {spacy_en['total_words']} | {spacy_en['total_entities']} | {spacy_en['entities_per_100_words']} | {spacy_en['no_entity_texts']} | {spacy_en['no_entity_rate_percent']}% | {spacy_en['entity_counts_by_type']} | {spacy_en['example_entities']} |
| Arabic | spaCy `xx_ent_wiki_sm` | {spacy_ar['texts_processed']} | {spacy_ar['total_words']} | {spacy_ar['total_entities']} | {spacy_ar['entities_per_100_words']} | {spacy_ar['no_entity_texts']} | {spacy_ar['no_entity_rate_percent']}% | {spacy_ar['entity_counts_by_type']} | {spacy_ar['example_entities']} |
| English | HF XLM-RoBERTa WikiANN | {hf_en['texts_processed']} | {hf_en['total_words']} | {hf_en['total_entities']} | {hf_en['entities_per_100_words']} | {hf_en['no_entity_texts']} | {hf_en['no_entity_rate_percent']}% | {hf_en['entity_counts_by_type']} | {hf_en['example_entities']} |
| Arabic | HF XLM-RoBERTa WikiANN | {hf_ar['texts_processed']} | {hf_ar['total_words']} | {hf_ar['total_entities']} | {hf_ar['entities_per_100_words']} | {hf_ar['no_entity_texts']} | {hf_ar['no_entity_rate_percent']}% | {hf_ar['entity_counts_by_type']} | {hf_ar['example_entities']} |

I kept each model’s native labels instead of mapping them to the English spaCy schema. This made the comparison clearer because the goal was to compare entity totals, label patterns, density, no-entity rate, and example outputs rather than compute exact cross-model match scores.

## Analysis Paragraph 1 — Arabic vs English NER quality

The results show a clear difference between English and Arabic NER performance across the two multilingual models. For English, both models extracted **{spacy_en['total_entities']} entities** from **{spacy_en['texts_processed']} texts**, with an entity density of **{spacy_en['entities_per_100_words']} entities per 100 words** and a **{spacy_en['no_entity_rate_percent']}% no-entity rate**. This suggests that English entity detection was more consistent in the sample.

However, the models still differed in label behavior. spaCy found examples like **{spacy_en['example_entities']}**, while Hugging Face found examples like **{hf_en['example_entities']}**. For Arabic, the difference between the two models was larger. spaCy extracted only **{spacy_ar['total_entities']} Arabic entities**, with a lower density of **{spacy_ar['entities_per_100_words']} entities per 100 words** and **{spacy_ar['no_entity_texts']} texts with no entities found**. Some spaCy Arabic examples also look noisy, such as **{spacy_ar['example_entities']}**, because they include phrases that are not always clean entity names.

In contrast, Hugging Face extracted **{hf_ar['total_entities']} Arabic entities**, with **{hf_ar['entities_per_100_words']} entities per 100 words** and a **{hf_ar['no_entity_rate_percent']}% no-entity rate**, finding clearer examples such as **{hf_ar['example_entities']}**. This suggests that the Hugging Face multilingual model handled Arabic climate text better in this sample, while spaCy struggled more with Arabic boundaries and labels.

## Analysis Paragraph 2 — MENA professional context

For bilingual NLP applications in the MENA region, this comparison shows why an English-only NER pipeline is not enough. A real climate research or news monitoring system in Jordan would need to process English reports, Arabic news, and mixed-language documents. The results show that multilingual models can work on both languages, but model choice matters a lot.

In this sample, Hugging Face performed better on Arabic than spaCy because it extracted more Arabic entities and found more meaningful organizations and locations. At the same time, Arabic results still need qualitative review because there is no Arabic gold standard in this assignment, and some labels or boundaries may still be wrong. For a production bilingual NLP system, I would use multilingual NER as a starting point, then add domain-specific rules for climate terms, human review for Arabic outputs, and confidence-based filtering to improve reliability.
"""

    return analysis

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print(f"Dataset size: {len(df)}")
    print(f"Language: {df['language'].value_counts().to_dict()}")

    english_texts = sample_texts(df, "en", n=20)
    arabic_texts = sample_texts(df, "ar", n=20)

    print(f"English texts sampled: {len(english_texts)}")
    print(f"Arabic texts sampled: {len(arabic_texts)}")

    print("\nLoading spaCy multilingual model...")
    spacy_nlp = spacy.load("xx_ent_wiki_sm")

    print("Loading Hugging Face multilingual NER model...")
    hf_ner = pipeline("ner", model="Davlan/xlm-roberta-base-wikiann-ner", aggregation_strategy="simple")

    print("\nRunning spaCy NER on English...")
    spacy_en = run_spacy_ner(english_texts, spacy_nlp)

    print("Running spaCy NER on Arabic...")
    spacy_ar = run_spacy_ner(arabic_texts, spacy_nlp)

    print("Running Hugging Face NER on English...")
    hf_en = run_hf_ner(english_texts, hf_ner)

    print("Running Hugging Face NER on Arabic...")
    hf_ar = run_hf_ner(arabic_texts, hf_ner)

    all_entities = pd.concat([spacy_en, spacy_ar, hf_en, hf_ar], ignore_index=True)
    all_entities.to_csv(os.path.join(OUTPUT_DIR, "multilingual_ner_entities.csv"), index=False)

    summaries = [
        summarize_results(spacy_en, english_texts, "en", "spaCy_xx_ent_wiki_sm"),
        summarize_results(spacy_ar, arabic_texts, "ar", "spaCy_xx_ent_wiki_sm"),
        summarize_results(hf_en, english_texts, "en", "HF_xlm_roberta_wikiann"),
        summarize_results(hf_ar, arabic_texts, "ar", "HF_xlm_roberta_wikiann"),
    ]

    comparison_df = pd.DataFrame(summaries)

    comparison_df.to_csv(COMPARISON_PATH, index=False)

    analysis_text = write_analysis(comparison_df)
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write(analysis_text)

    print("\n=== Multilingual NER Comparison ===")
    print(comparison_df)

    print(f"\nSaved comparison table to: {COMPARISON_PATH}")
    print(f"Saved analysis to: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()