"""
predict.py

Purpose:
    Load the trained model + vectorizer and predict sentiment for a
    single text input. Used by the Streamlit app.
"""

from __future__ import annotations

import os
import json
import pickle

import numpy as np

from tf_compat import require_tensorflow, tf

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "sentiment_model.keras")
VOCAB_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "vocabulary.json")
ENCODER_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "label_encoder.pkl")
MAX_LEN       = 50

SENTIMENT_EMOJI = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}


def rebuild_vectorizer(vocab_path: str = VOCAB_PATH):
    """
    Rebuild the TextVectorization layer from the saved vocabulary list.

    Why rebuild from vocabulary instead of saving/loading the whole layer?
    - TextVectorization doesn't load cleanly via tf.saved_model.load() in
      all TF versions. Saving the vocabulary list (a plain JSON array) and
      rebuilding the layer with set_vocabulary() is the most portable and
      version-robust approach — works identically across TF 2.x.
    - The vocabulary IS the only stateful part of the layer — rebuilding
      from it gives an identical transformation.
    """
    require_tensorflow()
    with open(vocab_path, "r") as f:
        vocabulary = json.load(f)
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(vocabulary),
        output_mode="int",
        output_sequence_length=MAX_LEN,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.set_vocabulary(vocabulary)
    return vectorizer


def load_artifacts():
    """Load model, rebuild vectorizer from vocab, and load label encoder."""
    require_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = rebuild_vectorizer()
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, vectorizer, le


def predict_sentiment(text: str, model=None, vectorizer=None, le=None) -> dict:
    if model is None or vectorizer is None or le is None:
        model, vectorizer, le = load_artifacts()

    text_vec = vectorizer([text])
    probs = model.predict(text_vec, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = le.classes_[pred_idx]

    return {
        "sentiment": label,
        "confidence": round(float(probs[pred_idx]), 4),
        "emoji": SENTIMENT_EMOJI[label],
        "all_probabilities": {
            le.classes_[i]: round(float(probs[i]), 4)
            for i in range(len(le.classes_))
        },
    }


if __name__ == "__main__":
    samples = [
        "This product is absolutely amazing love it",
        "Terrible quality broke immediately avoid",
        "It is okay nothing special does the job",
    ]
    model, vectorizer, le = load_artifacts()
    for text in samples:
        result = predict_sentiment(text, model, vectorizer, le)
        print(f"Text: {text!r}")
        print(f"  -> {result['emoji']} {result['sentiment']} ({result['confidence']:.1%})\n")
