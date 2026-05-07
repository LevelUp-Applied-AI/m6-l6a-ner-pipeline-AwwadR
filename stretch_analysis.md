# Stretch Analysis: Cross-Lingual Embedding Comparison

## Overview

This stretch assignment tested whether `bert-base-multilingual-cased` can place English and Arabic climate texts into a shared embedding space. I selected 10 English climate texts and 10 Arabic climate texts from `data/climate_articles.csv`, computed mean-pooled multilingual BERT embeddings, and created a 20x20 cosine similarity matrix.

The heatmap was saved to:

```text
outputs/cross_lingual_similarity_heatmap.png
```

## Results

The script successfully selected 10 English texts and 10 Arabic texts.

```text
English texts available: 132
Arabic texts available: 68
Selected English texts: 10
Selected Arabic texts: 10
Embedding matrix shape: (20, 768)
Similarity matrix shape: (20, 20)
```

Similarity summary:

```text
English-English mean similarity: 0.8038
Arabic-Arabic mean similarity:   0.8107
English-Arabic mean similarity:  0.6132
English-English max similarity:  0.8710
Arabic-Arabic max similarity:    0.9397
English-Arabic max similarity:   0.7516
English-Arabic min similarity:   0.4805
```

## Analysis

The multilingual BERT model showed partial cross-lingual alignment between English and Arabic climate texts. Same-language similarities were stronger overall, with English-English mean similarity at `0.8038` and Arabic-Arabic mean similarity at `0.8107`, while English-Arabic mean similarity was lower at `0.6132`. This shows that the model clusters texts more strongly within the same language, but it can still connect some related topics across English and Arabic. For example, the highest English-Arabic pair had a score of `0.7516`, connecting an English text about the IPCC Sixth Assessment Report with an Arabic text about COP28 and transitioning away from fossil fuels. Another strong pair scored `0.7375`, connecting the English IPCC report text with an Arabic text about the IPCC synthesis report. A lower cross-lingual score of `0.4805` shows that not all bilingual pairs are treated as equally similar, which means the model is capturing some topic differences rather than only language-level patterns.

For building bilingual NLP tools in the MENA region, these results suggest that multilingual BERT can be a useful starting point for Arabic-English climate search and retrieval, but it should not be used blindly in production. The model was able to connect some related climate topics across languages, such as IPCC reports, COP28, food security, and carbon border adjustment. However, the lower English-Arabic mean similarity compared with same-language similarity means bilingual retrieval quality may be weaker than English-only or Arabic-only retrieval. For a real MENA climate search system, I would test more paired examples, fine-tune on bilingual climate data, compare against stronger multilingual embedding models, and add filters for language, topic, and named entities to improve user trust and result quality.
