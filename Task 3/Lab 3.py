# =====================================================
# Task 3: Pattern Discovery from Cleaned YouTube Text
# =====================================================

import os
import ast
import itertools
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

nltk.download("stopwords")

# -----------------------------------------------------
# PATH SETUP
# -----------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMENTS_FILE = os.path.join(ROOT_DIR, "cleaned_comments.csv")
CAPTIONS_FILE = os.path.join(ROOT_DIR, "cleaned_captions.csv")

# -----------------------------------------------------
# A. LOAD & CLEAN DATA
# -----------------------------------------------------

def parse_tokens(value):
    try:
        tokens = ast.literal_eval(value)
        return tokens if isinstance(tokens, list) else []
    except:
        return []

comments_df = pd.read_csv(COMMENTS_FILE)
captions_df = pd.read_csv(CAPTIONS_FILE)

comments_df["cleaned_tokens"] = comments_df["cleaned_tokens"].apply(parse_tokens)
captions_df["cleaned_tokens"] = captions_df["cleaned_tokens"].apply(parse_tokens)

comments_df = comments_df[comments_df["cleaned_tokens"].apply(len) > 0]
captions_df = captions_df[captions_df["cleaned_tokens"].apply(len) > 0]

# -----------------------------------------------------
# UNIGRAM FREQUENCY
# -----------------------------------------------------

def plot_unigrams(df, title, filename):
    tokens = list(itertools.chain.from_iterable(df["cleaned_tokens"]))
    top = Counter(tokens).most_common(20)
    words, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.barh(words, counts)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_unigrams(comments_df, "Top 20 Unigrams (Comments)", "unigrams_comments.png")
plot_unigrams(captions_df, "Top 20 Unigrams (Captions)", "unigrams_captions.png")

# -----------------------------------------------------
# B. TRANSACTION CONSTRUCTION
# -----------------------------------------------------

def build_transactions(df):
    transactions = []
    transaction_ids = []

    for idx, tokens in enumerate(df["cleaned_tokens"]):
        unique_tokens = list(set(tokens))
        if len(unique_tokens) >= 3:
            transactions.append(unique_tokens)
            transaction_ids.append(f"TX_{idx}")

    return transactions, transaction_ids

comment_tx, comment_tx_ids = build_transactions(comments_df)
caption_tx, caption_tx_ids = build_transactions(captions_df)

lengths = [len(t) for t in comment_tx]

stats_df = pd.DataFrame({
    "average_length": [sum(lengths) / len(lengths)],
    "min_length": [min(lengths)],
    "max_length": [max(lengths)]
})
stats_df.to_csv(os.path.join(OUTPUT_DIR, "transaction_length_stats.csv"), index=False)

plt.hist(lengths, bins=20)
plt.xlabel("Basket Length")
plt.ylabel("Frequency")
plt.title("Transaction Length Distribution (Comments)")
plt.savefig(os.path.join(OUTPUT_DIR, "transaction_length_histogram.png"))
plt.close()

# -----------------------------------------------------
# C. MANUAL CO-OCCURRENCE ANALYSIS
# -----------------------------------------------------

stop_words = set(stopwords.words("english"))
pair_counter = Counter()

for basket in comment_tx:
    filtered = [w for w in basket if w not in stop_words]
    for pair in itertools.combinations(sorted(filtered), 2):
        pair_counter[pair] += 1

pairs = {p: c for p, c in pair_counter.items() if c >= 3}

cooc_df = pd.DataFrame(
    [(p[0], p[1], c) for p, c in pairs.items()],
    columns=["word1", "word2", "count"]
)
cooc_df.to_csv(os.path.join(OUTPUT_DIR, "cooccurrence_pairs.csv"), index=False)

top20 = cooc_df.sort_values("count", ascending=False).head(20)
plt.barh(top20["word1"] + " & " + top20["word2"], top20["count"])
plt.gca().invert_yaxis()
plt.title("Top 20 Co-occurring Word Pairs")
plt.savefig(os.path.join(OUTPUT_DIR, "cooccurrence_top20.png"))
plt.close()

G = nx.Graph()
for _, row in cooc_df.iterrows():
    G.add_edge(row["word1"], row["word2"], weight=row["count"])

plt.figure(figsize=(12, 10))
nx.draw(G, nx.spring_layout(G, k=0.4), with_labels=True, node_size=300, font_size=8)
plt.savefig(os.path.join(OUTPUT_DIR, "cooccurrence_network.png"))
plt.close()

# -----------------------------------------------------
# D. APRIORI ANALYSIS
# -----------------------------------------------------

def run_apriori(transactions, prefix):
    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(encoded, columns=te.columns_)

    supports = [0.3, 0.2, 0.1, 0.15, 0.05]
    all_itemsets = []
    all_rules = []

    for s in supports:
        fi = apriori(df, min_support=s, use_colnames=True)
        fi["support_level"] = s
        all_itemsets.append(fi)

        rules = association_rules(fi, metric="confidence", min_threshold=0.6)
        rules = rules[rules["lift"] >= 1.2]
        rules["support_level"] = s
        all_rules.append(rules)

    itemsets_df = pd.concat(all_itemsets, ignore_index=True)
    rules_df = pd.concat(all_rules, ignore_index=True)

    itemsets_df["length"] = itemsets_df["itemsets"].apply(len)
    itemsets_df = itemsets_df[itemsets_df["length"].isin([2, 3])]

    itemsets_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_frequent_itemsets.csv"), index=False)
    rules_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_association_rules.csv"), index=False)

    plt.scatter(rules_df["support"], rules_df["confidence"])
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title(f"Support vs Confidence ({prefix})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_support_confidence.png"))
    plt.close()

    return itemsets_df, rules_df

comment_itemsets, comment_rules = run_apriori(comment_tx, "comments")
caption_itemsets, caption_rules = run_apriori(caption_tx, "captions")
merged_itemsets, merged_rules = run_apriori(comment_tx + caption_tx, "merged")

# -----------------------------------------------------
# E. VARIATION EXPERIMENTS
# -----------------------------------------------------

stemmer = PorterStemmer()

def stem_and_filter(transactions):
    return [[stemmer.stem(w) for w in t if len(w) >= 4] for t in transactions]

stemmed_tx = stem_and_filter(comment_tx)
run_apriori(stemmed_tx, "comments_stemmed")

# -----------------------------------------------------
# F. INTERPRETATION & VISUALIZATION
# -----------------------------------------------------

top_2 = comment_itemsets[comment_itemsets["length"] == 2] \
    .sort_values("support", ascending=False).head(10)

plt.barh(top_2["itemsets"].astype(str), top_2["support"])
plt.gca().invert_yaxis()
plt.title("Top 10 2-Itemsets by Support")
plt.savefig(os.path.join(OUTPUT_DIR, "top_2_itemsets.png"))
plt.close()

top_rules = comment_rules.sort_values("confidence", ascending=False).head(10)

plt.barh(top_rules["antecedents"].astype(str), top_rules["confidence"])
plt.gca().invert_yaxis()
plt.title("Top 10 Rules by Confidence")
plt.savefig(os.path.join(OUTPUT_DIR, "top_rules_confidence.png"))
plt.close()

all_words = list(itertools.chain.from_iterable(comment_tx))
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(" ".join(all_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Frequent Tokens")
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
plt.close()

cluster_G = nx.Graph()
for _, row in cooc_df.iterrows():
    if row["count"] >= 5:
        cluster_G.add_edge(row["word1"], row["word2"], weight=row["count"])

plt.figure(figsize=(12, 10))
nx.draw(
    cluster_G,
    nx.spring_layout(cluster_G, k=0.5),
    with_labels=True,
    node_size=400,
    font_size=9
)
plt.title("Cluster Graph of Word Associations")
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_graph.png"))
plt.close()

print("Lab 3 completed successfully.")
print("All required outputs saved in the output folder.")
