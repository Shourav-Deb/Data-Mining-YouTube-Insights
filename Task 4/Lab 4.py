# ==========================================================
# Task 4 – Incremental Pattern Mining & Correlation Analysis
# ==========================================================


import pandas as pd
import numpy as np
import ast
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

# STEP 1: LOAD AND INSPECT DATA
df = pd.read_csv("cleaned_comments.csv")

print("Columns in dataset:", df.columns)

if "cleaned_tokens" not in df.columns:
    raise ValueError("cleaned_tokens column not found.")

if isinstance(df['cleaned_tokens'].iloc[0], str):
    df['cleaned_tokens'] = df['cleaned_tokens'].apply(ast.literal_eval)

print("Data successfully loaded and tokens verified.")
print("Total comments:", len(df))

# STEP 2: SPLIT INTO 5 CHUNKS
chunks = np.array_split(df.index, 5)
chunks = [df.loc[idx] for idx in chunks]

print("\nData split into", len(chunks), "chunks.")

all_unigram_counts = []
all_bigram_counts = []

# STEP 3: MINE FREQUENT PATTERNS PER CHUNK
for i, chunk in enumerate(chunks):
    print(f"\n======================")
    print(f"~> Processing Chunk {i+1}")
    print("======================")

    transactions = chunk['cleaned_tokens'].tolist()

    unigram_counts = Counter(
        token for row in transactions for token in row
    )
    top_unigrams = unigram_counts.most_common(10)

    print("Top 10 Unigrams:", top_unigrams)

    all_unigram_counts.append(unigram_counts)

    plt.figure()
    plt.bar([w for w, _ in top_unigrams],
            [c for _, c in top_unigrams])
    plt.title(f"Top 10 Unigrams - Chunk {i+1}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    bigram_counts = Counter()
    for row in transactions:
        unique_words = sorted(set(row))
        bigram_counts.update(combinations(unique_words, 2))

    top_bigrams = bigram_counts.most_common(10)
    print("Top 10 Co-occurring Pairs:", top_bigrams)

    all_bigram_counts.append(bigram_counts)

    bigram_labels = [f"{a},{b}" for (a, b), _ in top_bigrams]
    bigram_values = [c for _, c in top_bigrams]

    plt.figure()
    plt.bar(bigram_labels, bigram_values)
    plt.title(f"Top 10 Co-occurring Word Pairs - Chunk {i+1}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# STEP 4: TRACK PATTERNS ACROSS CHUNKS
total_unigram_counts = sum(all_unigram_counts, Counter())
selected_patterns = [word for word, _ in total_unigram_counts.most_common(5)]

print("\nSelected Patterns for Tracking:", selected_patterns)

freq_data = {}

for pattern in selected_patterns:
    freq_list = []
    for chunk_counts in all_unigram_counts:
        freq_list.append(chunk_counts.get(pattern, 0))
    freq_data[pattern] = freq_list

freq_df = pd.DataFrame(freq_data)
freq_df.index = [f"Chunk {i+1}" for i in range(5)]

print("\nFrequency Table Across Chunks:")
print(freq_df)

# STEP 5: CORRELATION ANALYSIS
corr = freq_df.corr()

print("\nCorrelation Matrix:")
print(corr)

# STEP 6: VISUALIZE PATTERN TRENDS
plt.figure()
freq_df.plot(marker='o')
plt.xlabel("Chunk")
plt.ylabel("Frequency")
plt.title("Pattern Frequency Over Time")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nLab 4 Cmplete.")