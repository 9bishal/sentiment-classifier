"""
train_model.py

Purpose:
    Build and train a text classification model: Embedding -> GlobalAveragePooling
    -> Dense layers. Evaluate with per-class metrics. Save the model.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tf_compat import require_tensorflow, tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "sentiment_model.keras")
PLOT_PATH     = os.path.join(PROJECT_ROOT, "images", "training_curves.png")
CM_PATH       = os.path.join(PROJECT_ROOT, "images", "confusion_matrix.png")

VOCAB_SIZE  = 5000
EMBED_DIM   = 32
NUM_CLASSES = 3


def build_model(vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM,
                num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Architecture: Embedding -> GlobalAveragePooling1D -> Dense -> Dropout -> Dense(softmax)

    Why this architecture?
    - Embedding layer: maps integer token IDs to dense vectors (trainable,
      learned during training). A 32-dimensional embedding is enough for a
      small vocabulary.
    - GlobalAveragePooling1D: averages the embeddings across all positions
      in the sequence. This converts a variable-length sequence of 32-dim
      vectors into a single fixed-size 32-dim vector representing the
      whole review. Simple and effective for sentiment classification,
      where position usually matters less than word presence.
    - Dense(64) + Dropout(0.4): a small hidden layer with dropout for
      regularisation.
    - Dense(3, softmax): output probabilities for the 3 classes.

    Why not an LSTM or Transformer (BERT)?
    - For 3-class sentiment on short reviews, a simple Embedding +
      pooling model performs surprisingly well and is fast to train.
    - LSTMs capture word ORDER — useful for tasks like machine translation
      or long document understanding. For short product reviews, word
      PRESENCE is often the dominant signal ("love", "terrible", "okay").
    - BERT would need pretrained weights from the internet (same download
      issue as MobileNetV2). Mentioning "I'd use BERT/DistilBERT for
      longer texts or when context matters more" is a strong interview
      talking point without adding unnecessary complexity.
    - This is a deliberate architecture tradeoff choice, not a limitation.
    """
    require_tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,  # +1 for the padding/OOV token at index 0
            output_dim=embed_dim,
            name="embedding",
        ),
        tf.keras.layers.GlobalAveragePooling1D(name="pooling"),
        tf.keras.layers.Dense(64, activation="relu", name="hidden"),
        tf.keras.layers.Dropout(0.4, name="dropout"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ])
    return model


def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def run_training():
    require_tensorflow()
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = le.classes_.tolist()

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # y labels are integers not one-hot
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    # Evaluation
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_history(history, PLOT_PATH)
    plot_confusion_matrix(y_test, y_pred, class_names, CM_PATH)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    run_training()
