# =====================================================
# Task 6: Text Mining with YouTube Data
# =====================================================


import os
import re
import ast
import sys
import subprocess
import importlib
from collections import Counter
from itertools import combinations


# -----------------------------------------------------
# Package Check
# -----------------------------------------------------

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "matplotlib_venn": "matplotlib-venn",
    "textblob": "textblob",
}


def ensure_package(module_name, package_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(module_name)


for module_name, package_name in REQUIRED_PACKAGES.items():
    ensure_package(module_name, package_name)


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib_venn import venn2
from textblob import TextBlob


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_KEYWORDS = 15
TOP_KEYWORDS_COMPARE = 20
TOP_BIGRAMS = 10
TOP_COOC_TERMS = 20
TOP_COOC_PAIRS = 15
TOP_TEMPORAL_TERMS = 5
TEMPORAL_CHUNKS = 5
RUN_OPTIONAL_CLUSTERING = False


# -----------------------------------------------------
# File Discovery
# -----------------------------------------------------


def find_file(target_name):
    candidates = []

    for root, _, files in os.walk(BASE_DIR):
        if os.path.abspath(root).startswith(os.path.abspath(OUTPUT_DIR)):
            continue

        for file_name in files:
            if file_name.lower() == target_name.lower():
                candidates.append(os.path.join(root, file_name))

    if not candidates:
        return None

    candidates.sort(key=lambda p: (p.count(os.sep), p.lower()))
    return candidates[0]


COMMENTS_FILE = find_file("cleaned_comments.csv")
CAPTIONS_FILE = find_file("cleaned_captions.csv")


# -----------------------------------------------------
# Data Helpers
# -----------------------------------------------------


def parse_token_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        return [token.strip() for token in value.split() if token.strip()]

    return []



def simple_clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if len(token) > 1]


# -----------------------------------------------------
# Part A - Setup and Recall
# -----------------------------------------------------


def load_comments_dataframe():
    if not COMMENTS_FILE or not os.path.exists(COMMENTS_FILE):
        raise FileNotFoundError(
            "cleaned_comments.csv not found. Put cleaned_comments.csv in the same folder as this script."
        )

    df = pd.read_csv(COMMENTS_FILE)

    if "cleaned_tokens" not in df.columns:
        if "comment_text" not in df.columns:
            raise ValueError("cleaned_comments.csv must contain cleaned_tokens or comment_text.")
        df["cleaned_tokens"] = df["comment_text"].fillna("").apply(simple_clean_text)
    else:
        df["cleaned_tokens"] = df["cleaned_tokens"].apply(parse_token_list)

    if "comment_text" not in df.columns:
        df["comment_text"] = df["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))

    df = df[df["cleaned_tokens"].map(len) > 0].copy()
    df["analysis_text"] = df["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))
    return df



def load_captions_dataframe():
    if not CAPTIONS_FILE or not os.path.exists(CAPTIONS_FILE):
        raise FileNotFoundError(
            "cleaned_captions.csv not found. Put cleaned_captions.csv in the same folder as this script."
        )

    df = pd.read_csv(CAPTIONS_FILE)

    if "cleaned_tokens" not in df.columns:
        if "caption_sentence" not in df.columns:
            raise ValueError("cleaned_captions.csv must contain cleaned_tokens or caption_sentence.")
        df["cleaned_tokens"] = df["caption_sentence"].fillna("").apply(simple_clean_text)
    else:
        df["cleaned_tokens"] = df["cleaned_tokens"].apply(parse_token_list)

    if "caption_sentence" not in df.columns:
        df["caption_sentence"] = df["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))

    df = df[df["cleaned_tokens"].map(len) > 0].copy()
    df["analysis_text"] = df["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))
    return df


comments_df = load_comments_dataframe()
captions_df = load_captions_dataframe()

comments_df.to_csv(os.path.join(OUTPUT_DIR, "lab6_comments_ready.csv"), index=False)
captions_df.to_csv(os.path.join(OUTPUT_DIR, "lab6_captions_ready.csv"), index=False)

print(f"\n# Comments source file: {COMMENTS_FILE}")
print(f"# Captions source file: {CAPTIONS_FILE}")
print(f"# Loaded {len(comments_df)} cleaned comment rows.")
print(f"# Loaded {len(captions_df)} cleaned caption rows.")


# -----------------------------------------------------
# TF-IDF Helper Functions
# -----------------------------------------------------


def build_tfidf_table(texts, top_n, min_df=2, max_df=0.85, ngram_range=(1, 1)):
    if len(texts) == 0:
        empty_df = pd.DataFrame(columns=["term", "mean_tfidf", "document_frequency"])
        return empty_df, empty_df

    try:
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(texts)

    terms = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()

    result_df = pd.DataFrame(
        {
            "term": terms,
            "mean_tfidf": mean_scores,
            "document_frequency": doc_freq,
        }
    )

    result_df = result_df.sort_values(
        by=["mean_tfidf", "document_frequency", "term"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top_df = result_df.head(top_n).copy().reset_index(drop=True)
    return top_df, result_df



def save_horizontal_bar(df, label_col, value_col, title, output_file, figsize=(10, 6)):
    if df.empty:
        return

    plot_df = df.iloc[::-1]
    plt.figure(figsize=figsize)
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel(value_col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------
# Part B - TF-IDF Keyword Extraction
# -----------------------------------------------------

comment_top_keywords_df, comment_all_keywords_df = build_tfidf_table(
    comments_df["analysis_text"].tolist(),
    top_n=TOP_KEYWORDS,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 1),
)

caption_top_keywords_df, caption_all_keywords_df = build_tfidf_table(
    captions_df["analysis_text"].tolist(),
    top_n=TOP_KEYWORDS,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 1),
)

comment_top_keywords_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_keywords_comments.csv"), index=False)
caption_top_keywords_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_keywords_captions.csv"), index=False)
comment_all_keywords_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_keywords_comments_all.csv"), index=False)
caption_all_keywords_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_keywords_captions_all.csv"), index=False)

save_horizontal_bar(
    comment_top_keywords_df,
    "term",
    "mean_tfidf",
    "Top TF-IDF Keywords - Comments",
    os.path.join(OUTPUT_DIR, "tfidf_keywords_comments.png"),
)

save_horizontal_bar(
    caption_top_keywords_df,
    "term",
    "mean_tfidf",
    "Top TF-IDF Keywords - Captions",
    os.path.join(OUTPUT_DIR, "tfidf_keywords_captions.png"),
)

print("\n# Top comment keywords:")
print(comment_top_keywords_df)

print("\n# Top caption keywords:")
print(caption_top_keywords_df)


# -----------------------------------------------------
# Part C - Keyword and Theme Comparison
# -----------------------------------------------------


def compare_terms(comment_terms, caption_terms):
    comment_set = set(comment_terms)
    caption_set = set(caption_terms)

    common_terms = sorted(comment_set & caption_set)
    comments_only = sorted(comment_set - caption_set)
    captions_only = sorted(caption_set - comment_set)

    rows = []
    for term in common_terms:
        rows.append({"keyword": term, "status": "common"})
    for term in comments_only:
        rows.append({"keyword": term, "status": "comments_only"})
    for term in captions_only:
        rows.append({"keyword": term, "status": "captions_only"})

    return pd.DataFrame(rows), common_terms, comments_only, captions_only


comment_compare_terms = comment_all_keywords_df["term"].head(TOP_KEYWORDS_COMPARE).tolist()
caption_compare_terms = caption_all_keywords_df["term"].head(TOP_KEYWORDS_COMPARE).tolist()

keyword_overlap_df, common_keywords, comment_only_keywords, caption_only_keywords = compare_terms(
    comment_compare_terms,
    caption_compare_terms,
)

keyword_overlap_df.to_csv(os.path.join(OUTPUT_DIR, "keyword_overlap_comparison.csv"), index=False)

plt.figure(figsize=(8, 6))
venn2([set(comment_compare_terms), set(caption_compare_terms)], set_labels=("Comments", "Captions"))
plt.title("Top 20 Keyword Overlap: Comments vs Captions")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "keyword_overlap_venn.png"), dpi=300, bbox_inches="tight")
plt.close()

with open(os.path.join(OUTPUT_DIR, "keyword_theme_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Common Keywords:\n")
    f.write(", ".join(common_keywords) if common_keywords else "None")
    f.write("\n\nComments Only Keywords:\n")
    f.write(", ".join(comment_only_keywords) if comment_only_keywords else "None")
    f.write("\n\nCaptions Only Keywords:\n")
    f.write(", ".join(caption_only_keywords) if caption_only_keywords else "None")


# -----------------------------------------------------
# Part D - N-gram Analysis (Bigrams)
# -----------------------------------------------------

comment_top_bigrams_df, comment_all_bigrams_df = build_tfidf_table(
    comments_df["analysis_text"].tolist(),
    top_n=TOP_BIGRAMS,
    min_df=2,
    max_df=0.85,
    ngram_range=(2, 2),
)

caption_top_bigrams_df, caption_all_bigrams_df = build_tfidf_table(
    captions_df["analysis_text"].tolist(),
    top_n=TOP_BIGRAMS,
    min_df=2,
    max_df=0.85,
    ngram_range=(2, 2),
)

comment_top_bigrams_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_bigrams_comments.csv"), index=False)
caption_top_bigrams_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_bigrams_captions.csv"), index=False)
comment_all_bigrams_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_bigrams_comments_all.csv"), index=False)
caption_all_bigrams_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_bigrams_captions_all.csv"), index=False)

save_horizontal_bar(
    comment_top_bigrams_df,
    "term",
    "mean_tfidf",
    "Top TF-IDF Bigrams - Comments",
    os.path.join(OUTPUT_DIR, "tfidf_bigrams_comments.png"),
    figsize=(11, 6),
)

save_horizontal_bar(
    caption_top_bigrams_df,
    "term",
    "mean_tfidf",
    "Top TF-IDF Bigrams - Captions",
    os.path.join(OUTPUT_DIR, "tfidf_bigrams_captions.png"),
    figsize=(11, 6),
)

bigram_overlap_df, common_bigrams, comment_only_bigrams, caption_only_bigrams = compare_terms(
    comment_top_bigrams_df["term"].tolist(),
    caption_top_bigrams_df["term"].tolist(),
)

bigram_overlap_df.to_csv(os.path.join(OUTPUT_DIR, "bigram_overlap_comparison.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "bigram_theme_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Common Bigrams:\n")
    f.write(", ".join(common_bigrams) if common_bigrams else "None")
    f.write("\n\nComments Only Bigrams:\n")
    f.write(", ".join(comment_only_bigrams) if comment_only_bigrams else "None")
    f.write("\n\nCaptions Only Bigrams:\n")
    f.write(", ".join(caption_only_bigrams) if caption_only_bigrams else "None")


# -----------------------------------------------------
# Part E - Sentiment Analysis
# -----------------------------------------------------


def label_sentiment(polarity):
    if polarity > 0.10:
        return "positive"
    if polarity < -0.10:
        return "negative"
    return "neutral"



def run_sentiment_analysis(df, text_column, prefix):
    result_df = df.copy()
    result_df["sentiment_polarity"] = result_df[text_column].fillna("").astype(str).apply(
        lambda text: TextBlob(text).sentiment.polarity
    )
    result_df["sentiment_label"] = result_df["sentiment_polarity"].apply(label_sentiment)

    summary_df = (
        result_df["sentiment_label"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
        .reset_index()
    )
    summary_df.columns = ["sentiment", "count"]
    summary_df["percentage"] = (summary_df["count"] / summary_df["count"].sum() * 100).round(2)

    result_df.to_csv(os.path.join(OUTPUT_DIR, f"sentiment_per_row_{prefix}.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"sentiment_summary_{prefix}.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_df["sentiment"], summary_df["count"])
    plt.ylabel("Count")
    plt.title(f"Sentiment Distribution - {prefix.title()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"sentiment_distribution_{prefix}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return result_df, summary_df


comments_sentiment_df, comments_sentiment_summary_df = run_sentiment_analysis(
    comments_df,
    "comment_text",
    "comments",
)

captions_sentiment_df, captions_sentiment_summary_df = run_sentiment_analysis(
    captions_df,
    "caption_sentence",
    "captions",
)

print("\n# Comment sentiment summary:")
print(comments_sentiment_summary_df)

print("\n# Caption sentiment summary:")
print(captions_sentiment_summary_df)


# -----------------------------------------------------
# Part F - Linking to Previous Labs
# -----------------------------------------------------


def find_cooccurring_pairs(df, allowed_terms, top_n=15):
    pair_counter = Counter()

    for tokens in df["cleaned_tokens"]:
        filtered_tokens = sorted(set(token for token in tokens if token in allowed_terms))
        for pair in combinations(filtered_tokens, 2):
            pair_counter[pair] += 1

    rows = []
    for (term_1, term_2), count in pair_counter.most_common(top_n):
        rows.append({"term_1": term_1, "term_2": term_2, "count": count})

    return pd.DataFrame(rows)



def save_pair_bar_chart(pair_df, title, output_file):
    if pair_df.empty:
        return

    plot_df = pair_df.copy()
    plot_df["pair"] = plot_df["term_1"] + " + " + plot_df["term_2"]
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["pair"], plot_df["count"])
    plt.xlabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


comment_top_term_set = set(comment_all_keywords_df["term"].head(TOP_COOC_TERMS).tolist())
caption_top_term_set = set(caption_all_keywords_df["term"].head(TOP_COOC_TERMS).tolist())

comment_pairs_df = find_cooccurring_pairs(comments_df, comment_top_term_set, top_n=TOP_COOC_PAIRS)
caption_pairs_df = find_cooccurring_pairs(captions_df, caption_top_term_set, top_n=TOP_COOC_PAIRS)

comment_pairs_df.to_csv(os.path.join(OUTPUT_DIR, "cooccurring_pairs_comments.csv"), index=False)
caption_pairs_df.to_csv(os.path.join(OUTPUT_DIR, "cooccurring_pairs_captions.csv"), index=False)

save_pair_bar_chart(
    comment_pairs_df,
    "Top Co-occurring Keyword Pairs - Comments",
    os.path.join(OUTPUT_DIR, "cooccurring_pairs_comments.png"),
)

save_pair_bar_chart(
    caption_pairs_df,
    "Top Co-occurring Keyword Pairs - Captions",
    os.path.join(OUTPUT_DIR, "cooccurring_pairs_captions.png"),
)



def run_temporal_analysis(df, tracked_terms, num_chunks=5):
    if df.empty:
        return pd.DataFrame()

    split_indexes = np.array_split(df.index.to_numpy(), num_chunks)
    chunk_rows = []

    for chunk_number, index_group in enumerate(split_indexes, start=1):
        if len(index_group) == 0:
            continue

        chunk_df = df.loc[index_group]
        token_counts = Counter(token for tokens in chunk_df["cleaned_tokens"] for token in tokens)

        row = {
            "chunk": f"Chunk {chunk_number}",
            "rows_in_chunk": len(chunk_df),
        }

        for term in tracked_terms:
            row[term] = token_counts.get(term, 0)

        chunk_rows.append(row)

    trend_df = pd.DataFrame(chunk_rows)
    trend_df.to_csv(os.path.join(OUTPUT_DIR, "temporal_keyword_trends_comments.csv"), index=False)

    if not trend_df.empty:
        plt.figure(figsize=(10, 6))
        for term in tracked_terms:
            plt.plot(trend_df["chunk"], trend_df[term], marker="o", label=term)
        plt.xlabel("Chunk")
        plt.ylabel("Frequency")
        plt.title("Temporal Keyword Trends in Comments")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "temporal_keyword_trends_comments.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return trend_df


tracked_terms = comment_all_keywords_df["term"].head(TOP_TEMPORAL_TERMS).tolist()
temporal_trends_df = run_temporal_analysis(comments_df, tracked_terms, num_chunks=TEMPORAL_CHUNKS)

if RUN_OPTIONAL_CLUSTERING:
    vectorizer = TfidfVectorizer(max_features=2000)
    matrix = vectorizer.fit_transform(comments_df["analysis_text"].tolist())

    if matrix.shape[0] >= 2:
        best_k = 4 if matrix.shape[0] >= 4 else 2
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)

        clustered_df = comments_df.copy()
        clustered_df["kmeans_cluster"] = labels
        clustered_df.to_csv(os.path.join(OUTPUT_DIR, "lab6_clustered_comments.csv"), index=False)

        terms = vectorizer.get_feature_names_out()
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]

        cluster_rows = []
        for cluster_id in range(best_k):
            top_terms = [terms[index] for index in order_centroids[cluster_id, :10]]
            cluster_rows.append(
                {
                    "cluster": cluster_id,
                    "size": int((labels == cluster_id).sum()),
                    "top_terms": ", ".join(top_terms),
                }
            )

        pd.DataFrame(cluster_rows).to_csv(os.path.join(OUTPUT_DIR, "lab6_cluster_top_terms.csv"), index=False)


# -----------------------------------------------------
# Part G - Insight Statements
# -----------------------------------------------------


def get_sentiment_percentage(summary_df, label_name):
    row = summary_df.loc[summary_df["sentiment"] == label_name, "percentage"]
    return float(row.iloc[0]) if not row.empty else 0.0


comment_positive = get_sentiment_percentage(comments_sentiment_summary_df, "positive")
comment_neutral = get_sentiment_percentage(comments_sentiment_summary_df, "neutral")
comment_negative = get_sentiment_percentage(comments_sentiment_summary_df, "negative")

caption_positive = get_sentiment_percentage(captions_sentiment_summary_df, "positive")
caption_neutral = get_sentiment_percentage(captions_sentiment_summary_df, "neutral")
caption_negative = get_sentiment_percentage(captions_sentiment_summary_df, "negative")

if not comment_pairs_df.empty:
    top_pair_text = (
        f"{comment_pairs_df.iloc[0]['term_1']} + {comment_pairs_df.iloc[0]['term_2']} "
        f"({int(comment_pairs_df.iloc[0]['count'])})"
    )
else:
    top_pair_text = "No strong co-occurring pair found"

if not temporal_trends_df.empty:
    term_columns = [col for col in temporal_trends_df.columns if col not in {"chunk", "rows_in_chunk"}]
    if term_columns:
        strongest_term = max(term_columns, key=lambda term: temporal_trends_df[term].max() - temporal_trends_df[term].min())
        temporal_text = f"Among the tracked comment keywords, '{strongest_term}' changed the most across the five chunks."
    else:
        temporal_text = "Temporal keyword variation could not be measured clearly."
else:
    temporal_text = "Temporal keyword variation could not be measured clearly."

insight_lines = [
    "1. Shared themes between captions and comments were: " + (", ".join(common_keywords[:5]) if common_keywords else "None") + ".",
    "2. Audience-focused themes appeared more clearly in comments through keywords such as: " + (", ".join(comment_only_keywords[:5]) if comment_only_keywords else "None") + ".",
    "3. Caption-focused themes appeared more clearly in spoken content through keywords such as: " + (", ".join(caption_only_keywords[:5]) if caption_only_keywords else "None") + ".",
    (
        "4. Sentiment differed between the two datasets. "
        f"Comments -> Positive: {comment_positive:.2f}%, Neutral: {comment_neutral:.2f}%, Negative: {comment_negative:.2f}%. "
        f"Captions -> Positive: {caption_positive:.2f}%, Neutral: {caption_neutral:.2f}%, Negative: {caption_negative:.2f}%."
    ),
    "5. The strongest comment co-occurring keyword pair was: " + top_pair_text + ". " + temporal_text,
]

with open(os.path.join(OUTPUT_DIR, "Lab6_Insights.txt"), "w", encoding="utf-8") as f:
    for line in insight_lines:
        f.write(line + "\n")


print("\n-> Lab 6 analysis complete.")
