# Stretch 6A — Multilingual NER Comparison Analysis

## Results Summary

I processed **20 English texts** and **20 Arabic texts** from the climate articles dataset using two multilingual NER models: spaCy `xx_ent_wiki_sm` and Hugging Face `Davlan/xlm-roberta-base-wikiann-ner`.

| Language | Model | Texts | Words | Total entities | Entities / 100 words | No-entity texts | No-entity rate | Entity counts | Example entities |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| English | spaCy `xx_ent_wiki_sm` | 20 | 1221 | 93 | 7.62 | 0 | 0.0% | {'ORG': 39, 'LOC': 27, 'MISC': 15, 'PER': 12} | IPCC (MISC); Sixth Assessment Report (MISC); Celsius (PER) |
| Arabic | spaCy `xx_ent_wiki_sm` | 20 | 930 | 20 | 2.15 | 5 | 25.0% | {'PER': 9, 'MISC': 7, 'ORG': 2, 'LOC': 2} | وأكد التقرير (PER); وقّع الأردن (MISC); وأكد وزير (PER) |
| English | HF XLM-RoBERTa WikiANN | 20 | 1221 | 93 | 7.62 | 0 | 0.0% | {'ORG': 55, 'LOC': 28, 'PER': 10} | Antonio Guterres (PER); COP (ORG); COP (ORG) |
| Arabic | HF XLM-RoBERTa WikiANN | 20 | 930 | 64 | 6.88 | 0 | 0.0% | {'ORG': 33, 'LOC': 29, 'PER': 2} | الهيئة الحكومية الدولية المعنية بتغير المناخ (ORG); الأردن (LOC); البنك الدولي (ORG) |

I kept each model’s native labels instead of mapping them to the English spaCy schema. This made the comparison clearer because the goal was to compare entity totals, label patterns, density, no-entity rate, and example outputs rather than compute exact cross-model match scores.

## Analysis Paragraph 1 — Arabic vs English NER quality

The results show a clear difference between English and Arabic NER performance across the two multilingual models. For English, both models extracted the same total number of entities: **93 entities** from **20 texts**, with an entity density of **7.62 entities per 100 words** and a **0.0% no-entity rate**. This suggests that English entity detection was more consistent in the sample.

However, the models still differed in label behavior. spaCy found examples like **IPCC (MISC); Sixth Assessment Report (MISC); Celsius (PER)**, while Hugging Face found examples like **Antonio Guterres (PER); COP (ORG); COP (ORG)**. For Arabic, the difference between the two models was larger. spaCy extracted only **20 Arabic entities**, with a lower density of **2.15 entities per 100 words** and **5 texts with no entities found**. Some spaCy Arabic examples also look noisy, such as **وأكد التقرير (PER); وقّع الأردن (MISC); وأكد وزير (PER)**, because they include phrases that are not always clean entity names.

In contrast, Hugging Face extracted **64 Arabic entities**, with **6.88 entities per 100 words** and a **0.0% no-entity rate**, finding clearer examples such as **الهيئة الحكومية الدولية المعنية بتغير المناخ (ORG); الأردن (LOC); البنك الدولي (ORG)**. This suggests that the Hugging Face multilingual model handled Arabic climate text better in this sample, while spaCy struggled more with Arabic boundaries and labels.

## Analysis Paragraph 2 — MENA professional context

For bilingual NLP applications in the MENA region, this comparison shows why an English-only NER pipeline is not enough. A real climate research or news monitoring system in Jordan would need to process English reports, Arabic news, and mixed-language documents. The results show that multilingual models can work on both languages, but model choice matters a lot.

In this sample, Hugging Face performed better on Arabic than spaCy because it extracted more Arabic entities and found more meaningful organizations and locations. At the same time, Arabic results still need qualitative review because there is no Arabic gold standard in this assignment, and some labels or boundaries may still be wrong. For a production bilingual NLP system, I would use multilingual NER as a starting point, then add domain-specific rules for climate terms, human review for Arabic outputs, and confidence-based filtering to improve reliability.
