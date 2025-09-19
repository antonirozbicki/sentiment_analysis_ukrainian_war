# Ukraine War Tweet Sentiment - EN (DistilBERT vs CardiffNLP)

This repo contains an end-to-end sentiment analysis pipeline on recent English tweets about the war in Ukraine.  
It demonstrates a practical NLP workflow: data collection (notebook), text preprocessing, model-based sentiment
classification (transformers), model comparison, and clean visualizations.

## Project goals
- Build a reproducible baseline for tweet sentiment (negative / neutral / positive).
- Compare two models: DistilBERT (2 classes) vs CardiffNLP (3 classes).
- Summarize agreement (incl. Cohen’s κ) and produce publication-ready charts.

## Repository structure
```
.
├─ notebooks/
│  └─ Sentiment_analysis_war.ipynb     # E2E analysis, figures, interpretation
├─ main.py                              # One-file pipeline on an existing CSV
├─ reports/                             # Outputs (figures, metrics, scored CSV)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> **Data note:** The pipeline expects a CSV with at least a `text` column (and optionally `created_at`).
> Tweet collection (via Tweepy / X API) is shown in the notebook.

## Quickstart
1) Install deps:
```bash
pip install -r requirements.txt
```

2) Prepare input CSV  
Ensure a file like `tweets_ukraine_en.csv` exists with a column `text` (and ideally `created_at`).

3) Run the pipeline:
```bash
python main.py --input_csv tweets_ukraine_en.csv --output_csv reports/tweets_scored.csv --out_dir reports
```

Artifacts produced:
- `reports/figures/distil_distribution.png`  
- `reports/figures/cardiff_distribution.png`  
- `reports/figures/agreement_heatmap.png`  
- `reports/metrics.txt` (agreement % and Cohen’s κ)  
- `reports/tweets_scored.csv` (predictions + confidence scores)

## Models
- **DistilBERT (sst-2)**: binary sentiment (`negative` / `positive`)  
- **CardiffNLP twitter-roberta-base-sentiment-latest**: 3-class sentiment (`negative` / `neutral` / `positive`)

## Results (example, will vary)
- High agreement on clearly polar tweets; Cohen’s κ typically in the “substantial” to “almost perfect” range.
- CardiffNLP is more conservative and often assigns *neutral* where DistilBERT picks a side.

## Limitations
- Small sample size (API limits, recent tweets only).
- English-only baseline (no geolocation, no bot filtering).
- Model bias and potential misread of sarcasm/factual news style.

## Next steps
- Increase sample size; add multilingual models (PL/UA/RU).
- Confidence filtering (e.g., ≥ 0.75) for more reliable summaries.
- Time-based trends with meaningful aggregation windows (weekly).
- Optional: collection script (`tweepy`) + `.env` for X API token.
