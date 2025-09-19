import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from transformers import pipeline

# Text preprocessing
def clean_tweet(text: str) -> str:
    s = str(text)
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+|#\w+", " ", s)
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# Plot helpers
def plot_distribution(series, order, title, outpath):
    pct = series.value_counts(normalize=True).reindex(order, fill_value=0).mul(100)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(pct.index, pct.values)
    plt.ylim(0, max(100, pct.max() + 10))
    plt.title(title)
    plt.xlabel("class")
    plt.ylabel("percentage (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, pct.values):
        plt.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_agreement_heatmap(df, outpath):
    distil_order  = ["negative", "positive"]              # rows
    cardiff_order = ["negative", "neutral", "positive"]   # cols
    ct = pd.crosstab(df["sentiment"], df["sentiment_cardiff"])\
           .reindex(index=distil_order, columns=cardiff_order, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)

    plt.figure(figsize=(7, 4.6))
    im = plt.imshow(ct_pct.values, aspect="auto", vmin=0, vmax=100)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("row %")
    plt.xticks(np.arange(len(cardiff_order)), cardiff_order)
    plt.yticks(np.arange(len(distil_order)), distil_order)
    plt.title("Model agreement (row-normalized %)")

    for i in range(ct_pct.shape[0]):
        for j in range(ct_pct.shape[1]):
            val = ct_pct.iat[i, j]
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center",
                     color="white" if val > 55 else "black", fontsize=9)
    plt.xlabel("CardiffNLP")
    plt.ylabel("DistilBERT")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Main pipeline
def run(input_csv: str, output_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(input_csv)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # 2) Preprocess
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].map(clean_tweet)

    # 3) Models
    distil = pipeline("sentiment-analysis")
    cardiff = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    # DistilBERT (2 classes)
    distil_out = df["clean_text"].map(lambda x: distil(x)[0])
    df["sentiment"] = distil_out.map(lambda r: r["label"].lower())
    df["score"]     = distil_out.map(lambda r: r["score"])

    # CardiffNLP (3 classes)
    cardiff_out = df["clean_text"].map(lambda x: cardiff(x)[0])
    df["sentiment_cardiff"] = cardiff_out.map(lambda r: r["label"].lower())
    df["score_cardiff"]     = cardiff_out.map(lambda r: r["score"])

    # 4) Save scored data
    df.to_csv(output_csv, index=False)

    # 5) Distributions
    plot_distribution(
        df["sentiment"], ["negative", "positive"],
        "DistilBERT sentiment distribution (2 classes)",
        os.path.join(fig_dir, "distil_distribution.png"),
    )
    plot_distribution(
        df["sentiment_cardiff"], ["negative", "neutral", "positive"],
        "CardiffNLP sentiment distribution (3 classes)",
        os.path.join(fig_dir, "cardiff_distribution.png"),
    )

    # 6) Agreement heatmap
    plot_agreement_heatmap(df, os.path.join(fig_dir, "agreement_heatmap.png"))

    # 7) Agreement metrics (binary; exclude neutral)
    bin_df = df[df["sentiment_cardiff"].isin(["negative", "positive"])].copy()
    agreement = (bin_df["sentiment"] == bin_df["sentiment_cardiff"]).mean() if len(bin_df) else np.nan
    kappa = cohen_kappa_score(bin_df["sentiment"], bin_df["sentiment_cardiff"]) if len(bin_df) else np.nan

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Binary agreement (no 'neutral'): {agreement:.3f}\n")
        f.write(f"Cohen's kappa: {kappa:.3f}\n")

    print("Done.")
    print(f"- Scored CSV: {output_csv}")
    print(f"- Figures: {fig_dir}/distil_distribution.png, {fig_dir}/cardiff_distribution.png, {fig_dir}/agreement_heatmap.png")
    print(f"- Metrics: {os.path.join(out_dir, 'metrics.txt')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end sentiment analysis on tweets CSV (EN).")
    ap.add_argument("--input_csv",  default="tweets_ukraine_en.csv", help="Path to input CSV with columns: text, created_at, ...")
    ap.add_argument("--output_csv", default="tweets_scored.csv",     help="Output CSV with model scores.")
    ap.add_argument("--out_dir",    default="reports",               help="Output directory for figures and metrics.")
    args = ap.parse_args()
    run(args.input_csv, args.output_csv, args.out_dir)
