import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

STANDARD_LABELS = {"ORG", "GPE", "DATE", "LAW", "MONEY", "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"}

def load_data():
    articles = pd.read_csv("data/climate_articles.csv")
    gold = pd.read_csv("data/gold_entities.csv")
    return articles, gold

def get_climate_patterns():
    return [
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},
        {"label": "CLIMATE_EVENT", "pattern": "UN Climate Change Conference"},

        {"label": "AGREEMENT", "pattern": "Paris Agreement"},
        {"label": "AGREEMENT", "pattern": "Kyoto Protocol"},

        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "REPORT", "pattern": "Global Stocktake"},

        {"label": "THRESHOLD", "pattern": "1.5°C"},
        {"label": "THRESHOLD", "pattern": "2°C target"},
        {"label": "THRESHOLD", "pattern": "net zero"}
    ]

def build_nlp_with_ruler(position="before"):
    nlp = spacy.load("en_core_web_sm")

    if position == "before":
        ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    else:
        ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": False})

    ruler.add_patterns(get_climate_patterns())
    
    return nlp

def extract_entities(df, nlp):
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
                "end_char": ent.end_char,
                "category": row["category"]
            })
    return pd.DataFrame(entities, columns=[
        "text_id", "entity_text", "entity_label", "start_char", "end_char", "category"
    ])

def compare_counts(base_df, before_df, after_df):
    comparison = pd.DataFrame({
        "base_spacy": base_df["entity_label"].value_counts(),
        "ruler_before_ner": before_df["entity_label"].value_counts(),
        "ruler_after_ner": after_df["entity_label"].value_counts()
    }).fillna(0).astype(int)

    return comparison

def evaluate_standard_labels(predicted_df, gold_df):
    predicted_standard = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)]
    gold_standard = gold_df[gold_df["entity_label"].isin(STANDARD_LABELS)]

    predicted_set = set(zip(
        predicted_standard["text_id"],
        predicted_standard["entity_text"],
        predicted_standard["entity_label"]
    ))

    gold_set = set(zip(
        gold_standard["text_id"],
        gold_standard["entity_text"],
        gold_standard["entity_label"]
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

def get_custom_rule_examples(entities_df):
    custom_labels = {"CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD", "AGREEMENT"}
    examples = entities_df[entities_df["entity_label"].isin(custom_labels)][["text_id", "category", "entity_text", "entity_label"]]

    return examples.drop_duplicates().head(20)

def save_outputs(counts_df, metrics_df, examples_df):
    counts_df.to_csv("stretch_entity_count_comparison.csv")
    metrics_df.to_csv("stretch_evaluation_delta.csv", index=False)
    examples_df.to_csv("stretch_custom_rule_examples.csv", index=False)

if __name__ == "__main__":
    articles, gold = load_data()

    print("Loading base spaCy...")
    base_nlp = spacy.load("en_core_web_sm")

    print("Loading spaCy with EntityRuler BEFORE NER...")
    before_nlp = build_nlp_with_ruler(position="before")

    print("Loading spaCy with EntityRuler AFTER NER...")
    after_nlp = build_nlp_with_ruler(position="after")

    print("Extracting base spaCy entities...")
    base_entities = extract_entities(articles, base_nlp)

    print("Extracting entities with ruler before NER...")
    before_entities = extract_entities(articles, before_nlp)

    print("Extracting entities with ruler after NER...")
    after_entities = extract_entities(articles, after_nlp)

    print("\nEntity count comparison:")
    counts = compare_counts(base_entities, before_entities, after_entities)
    print(counts)

    print("\nEvaluating standard labels only:")
    base_metrics = evaluate_standard_labels(base_entities, gold)
    before_metrics = evaluate_standard_labels(before_entities, gold)
    after_metrics = evaluate_standard_labels(after_entities, gold)

    metrics_df = pd.DataFrame([
        {"system": "base_spacy", **base_metrics},
        {"system": "ruler_before_ner", **before_metrics},
        {"system": "ruler_after_ner", **after_metrics},
    ])

    print(metrics_df)

    print("\nCustom rule examples:")
    examples = get_custom_rule_examples(before_entities)
    print(examples)

    save_outputs(counts, metrics_df, examples)

    print("\nSaved outputs:")
    print("- stretch_entity_count_comparison.csv")
    print("- stretch_evaluation_delta.csv")
    print("- stretch_custom_rule_examples.csv")