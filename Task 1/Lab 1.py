# =====================================================
# Task 1: Exploratory Data Analysis on YouTube Text Data
# =====================================================


import os
import matplotlib.pyplot as plt
from collections import Counter
import webvtt

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMENTS_FILE = os.path.join(BASE_DIR, "comments.txt")
CAPTIONS_FILE = os.path.join(BASE_DIR, "captions.vtt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------
# A. Loading Comments and Captions
# -----------------------------------------------------

def load_raw_comments(filepath="comments.txt"):
    comments = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if (
                len(line) > 0
                and not line.lower().endswith("ago")
                and line.lower() != "reply"
            ):
                comments.append(line)
    return comments


def load_vtt_captions(filepath="captions.vtt"):
    captions = []
    for caption in webvtt.read(filepath):
        text = caption.text.replace("\n", " ").strip()
        if text:
            captions.append(text)
    return captions


raw_comments = load_raw_comments(COMMENTS_FILE)
raw_captions = load_vtt_captions(CAPTIONS_FILE)

print(f"\n# Loaded {len(raw_comments)} potential comment lines.")
print(raw_comments[:5])

print(f"\n# Loaded {len(raw_captions)} caption lines.")
print(raw_captions[:5])

# -----------------------------------------------------
# B. Required Lab Experiments
# -----------------------------------------------------

# -------------------------
# 1. Histogram of Lengths
# -------------------------

comment_lengths = [len(c) for c in raw_comments]      # character length
caption_lengths = [len(c) for c in raw_captions]      # character length

plt.figure(figsize=(8, 5))
plt.hist(caption_lengths, bins=20, alpha=0.7, label="Captions")
plt.hist(comment_lengths, bins=20, alpha=0.7, label="Comments")
plt.xlabel("Length (characters)")
plt.ylabel("Frequency")
plt.title("Caption vs. Comment Lengths")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "length_histogram.png"))
plt.close()

# -------------------------
# 2. Vocabulary Diversity
#    (Type-Token Ratio)
# -------------------------

def type_token_ratio(lines):
    words = [word.lower() for line in lines for word in line.split()]
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0


caption_ttr = type_token_ratio(raw_captions)
comment_ttr = type_token_ratio(raw_comments)

print("\n> Caption TTR:", caption_ttr)
print("> Comment TTR:", comment_ttr)

# -------------------------
# 3. Top-N Word Frequency
# (After Stopword Removal)
# -------------------------

stopwords = set([
    "the", "and", "a", "is", "in", "to", "of", "that", "it",
    "on", "for", "with", "as", "this", "was", "but", "are",
    "not", "be", "at", "by", "an", "if", "or", "from", "so", "we"
])

def top_n_words(lines, n=20):
    words = [
        word.lower()
        for line in lines
        for word in line.split()
        if word.lower() not in stopwords
    ]
    return Counter(words).most_common(n)


top_caption_words = top_n_words(raw_captions, 20)
top_comment_words = top_n_words(raw_comments, 20)

print("\n# Top 20 caption words:\n", top_caption_words)
print("\n# Top 20 comment words:\n", top_comment_words)

# -----------------------------------------------------
# Save word frequency results
# -----------------------------------------------------

with open(os.path.join(OUTPUT_DIR, "top_words.txt"), "w", encoding="utf-8") as f:
    f.write("Top 20 Caption Words:\n")
    for word, count in top_caption_words:
        f.write(f"{word}: {count}\n")

    f.write("\nTop 20 Comment Words:\n")
    for word, count in top_comment_words:
        f.write(f"{word}: {count}\n")

print("\n-> Lab 1 analysis complete.")
