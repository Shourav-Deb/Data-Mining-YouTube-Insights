# =====================================================
# Task 2: Data Cleaning & Preprocessing for YouTube Text
# =====================================================

import re
import os
import pandas as pd
import webvtt

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMENTS_FILE = os.path.join(BASE_DIR, "comments.txt")
CAPTIONS_FILE = os.path.join(BASE_DIR, "captions.vtt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =====================================================
# 1. PARSE COMMENTS.TXT
# =====================================================

def parse_comments(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    comments = []
    time_pattern = re.compile(
        r".*(second|minute|hour|day|week|month|year)s?\s+ago.*",
        re.IGNORECASE
    )

    i = 0
    while i < len(lines) - 1:
        if time_pattern.match(lines[i + 1]):
            username = lines[i]
            timestamp = lines[i + 1]
            i += 2

            comment_text = []
            while i < len(lines) and not (
                i + 1 < len(lines) and time_pattern.match(lines[i + 1])
            ):
                if lines[i].lower() not in {"reply", "...more"}:
                    comment_text.append(lines[i])
                i += 1

            if comment_text:
                comments.append({
                    "username": username,
                    "timestamp_text": timestamp,
                    "comment_text": " ".join(comment_text)
                })
        else:
            i += 1

    return pd.DataFrame(comments)

# =====================================================
# 2. PARSE CAPTIONS.VTT
# =====================================================

def parse_captions(filepath):
    raw_lines = []

    try:
        for caption in webvtt.read(filepath):
            text = caption.text.replace("\n", " ").strip()
            raw_lines.append(text)
    except Exception:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if (
                    line
                    and "-->" not in line
                    and not line.isdigit()
                    and "WEBVTT" not in line
                ):
                    raw_lines.append(line)
    cleaned = []
    for line in raw_lines:
        line = re.sub(r"<c[^>]*>", "", line)
        line = re.sub(r"</c>", "", line)
        line = re.sub(r"<\d+:\d+:\d+\.\d+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)

    deduped = []
    for line in cleaned:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    sentences = []
    buffer = []

    for line in deduped:
        buffer.append(line)
        if re.search(r"[.!?]$", line) or len(" ".join(buffer).split()) >= 25:
            sentences.append(" ".join(buffer))
            buffer = []

    if buffer:
        sentences.append(" ".join(buffer))

    return pd.DataFrame({"caption_sentence": sentences})

# =====================================================
# 3. CLEANING PIPELINE
# =====================================================

def clean_text_pipeline(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()


    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens

# =====================================================
# 4. APPLY PIPELINE
# =====================================================

comments_df = parse_comments(COMMENTS_FILE)
captions_df = parse_captions(CAPTIONS_FILE)

comments_df["cleaned_tokens"] = comments_df["comment_text"].apply(clean_text_pipeline)
captions_df["cleaned_tokens"] = captions_df["caption_sentence"].apply(clean_text_pipeline)

comments_df = comments_df[comments_df["cleaned_tokens"].map(len) > 0]
captions_df = captions_df[captions_df["cleaned_tokens"].map(len) > 0]

# =====================================================
# 5. EXPORT CLEANED DATA
# =====================================================

comments_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_comments.csv"), index=False)
captions_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_captions.csv"), index=False)

print("Lab 2 complete.")
print(f"Cleaned comments rows: {len(comments_df)}")
print(f"Cleaned captions rows: {len(captions_df)}")
