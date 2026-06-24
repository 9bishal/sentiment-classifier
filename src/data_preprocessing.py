"""
data_preprocessing.py

Purpose:
    Load the raw reviews CSV, clean the text, encode labels, split into
    train/test, tokenize using TensorFlow's TextVectorization layer, and
    save the processed data.

Run:
    python src/data_preprocessing.py
"""

from __future__ import annotations

import os
import shutil
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tf_compat import require_tensorflow, tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH     = os.path.join(PROJECT_ROOT, "data", "raw", "reviews.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VOCAB_SIZE   = 5000    # maximum vocabulary size
MAX_LEN      = 50      # pad/truncate all sequences to this length
RAW_CANDIDATES = [
    RAW_PATH,
    os.path.join(PROJECT_ROOT, "data", "raw", "train.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "test.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "testdata.manual.2009.06.14.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "training.1600000.processed.noemoticon.csv"),
]


def reset_processed_dir() -> None:
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def resolve_raw_path() -> str:
    for path in RAW_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No supported raw sentiment CSV found in data/raw/. "
        "Expected one of: reviews.csv, train.csv, test.csv, "
        "testdata.manual.2009.06.14.csv, training.1600000.processed.noemoticon.csv"
    )


def clean_text(text: str) -> str:
    """
    Lowercase and strip leading/trailing whitespace.

    Why no stemming or stopword removal?
    - Sentiment is heavily carried by words like "not", "never", "very",
      "extremely" — removing stopwords would destroy this signal.
    - TensorFlow's TextVectorization handles lowercasing and basic
      punctuation removal via its standardize parameter (default:
      'lower_and_strip_punctuation'), so we let the layer handle it
      rather than duplicating logic here.
    """
    return str(text).lower().strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clean_cols = {col: col.replace("\ufffd", "").replace("�", "") for col in df.columns}
    columns = set(clean_cols.values())
    if {"text", "label"}.issubset(columns):
        return df[["text", "label"]]
    if {"text", "sentiment"}.issubset(columns):
        return df.rename(columns={"sentiment": "label"})[["text", "label"]]
    if {"text of the tweet", "polarity of tweet"}.issubset(columns):
        df = df.rename(columns={clean_cols["text of the tweet"]: "text", clean_cols["polarity of tweet"]: "label"})
        df["label"] = df["label"].map({0: "negative", 4: "positive"})
        return df[["text", "label"]]
    raise ValueError(f"Unsupported sentiment CSV columns: {sorted(df.columns)}")


def read_csv_flex(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="replace")


def encode_labels(df: pd.DataFrame):
    """
    Convert string labels (negative/neutral/positive) to integers (0/1/2).
    Saves the encoder so predict.py can map predictions back to strings.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    encoder_path = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Label classes: {le.classes_}")
    return y, le


def build_vectorizer(texts, vocab_size: int = VOCAB_SIZE, max_len: int = MAX_LEN):
    """
    Fit a TextVectorization layer on the training texts.

    TextVectorization does:
    1. Lowercases and strips punctuation (standardize='lower_and_strip_punctuation')
    2. Splits on whitespace
    3. Maps each token to an integer index (top vocab_size tokens by frequency)
    4. Pads/truncates sequences to max_len

    Why save the vectorizer separately (not bake into the model)?
    - predict.py needs to apply the SAME vocabulary mapping to new text
      at inference time. Saving it as a standalone layer ensures the
      vocabulary is identical between training and serving.
    - Alternative: include TextVectorization as the first layer in the
      Keras model (also valid, and slightly more elegant). Both approaches
      are worth knowing and comparing in an interview.
    """
    require_tensorflow()
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(texts)

    # Save vocabulary (the important state) as a plain list so predict.py
    # can rebuild an identical vectorizer without needing saved_model.load()
    import json
    vocab_path = os.path.join(PROCESSED_DIR, "vocabulary.json")
    with open(vocab_path, "w") as f:
        json.dump(vectorizer.get_vocabulary(), f)
    print(f"Vocabulary saved. Size: {len(vectorizer.get_vocabulary())}")
    return vectorizer


def run_pipeline():
    require_tensorflow()
    reset_processed_dir()

    raw_path = resolve_raw_path()
    df = read_csv_flex(raw_path)
    df = normalize_columns(df)
    df["text"] = df["text"].apply(clean_text)
    print(f"Loaded {len(df)} rows from {os.path.basename(raw_path)}. Label distribution:\n{df['label'].value_counts()}")

    y, le = encode_labels(df)

    # Stratified split to keep class balance in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = np.array(X_train), np.array(X_test)

    # Fit vectorizer on TRAINING text only — never on test text
    # (using test vocabulary would be a form of data leakage)
    vectorizer = build_vectorizer(X_train)

    X_train_vec = vectorizer(X_train).numpy()
    X_test_vec  = vectorizer(X_test).numpy()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_vec)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test_vec)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    print(f"Train: {X_train_vec.shape}, Test: {X_test_vec.shape}")
    print("Preprocessing complete.")
    return X_train_vec, X_test_vec, y_train, y_test, le


if __name__ == "__main__":
    run_pipeline()
