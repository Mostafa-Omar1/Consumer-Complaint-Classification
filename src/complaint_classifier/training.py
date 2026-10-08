"""Training and evaluation entry point for recurrent and transformer models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from complaint_classifier.data import load_dataset, stratified_split


RECURRENT_MODELS = ("rnn", "lstm", "gru")
ALL_MODELS = (*RECURRENT_MODELS, "transformer")


def _save_metrics(
    model_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_dir: Path,
) -> dict[str, float]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classes": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _save_confusion_matrix(model_name, matrix, class_names, model_dir)
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def _save_confusion_matrix(
    model_name: str,
    matrix: np.ndarray,
    class_names: Sequence[str],
    model_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{model_name.upper()} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(model_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def _fit_recurrent_model(
    model_name: str,
    *,
    train_texts: list[str],
    validation_texts: list[str],
    test_texts: list[str],
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    class_names: list[str],
    class_weights: dict[int, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[float, dict[str, float]]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=args.max_vocab_size,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(train_texts)

    def encode(texts: list[str]) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(texts)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=args.max_length,
            padding="post",
            truncating="post",
        )

    x_train = encode(train_texts)
    x_validation = encode(validation_texts)
    x_test = encode(test_texts)
    vocab_size = min(args.max_vocab_size, len(tokenizer.word_index) + 1)

    inputs = tf.keras.Input(shape=(args.max_length,), dtype="int32")
    embedded = tf.keras.layers.Embedding(vocab_size, args.embedding_dim)(inputs)
    if model_name == "rnn":
        encoded = tf.keras.layers.SimpleRNN(64)(embedded)
    elif model_name == "lstm":
        encoded = tf.keras.layers.LSTM(64)(embedded)
    else:
        encoded = tf.keras.layers.GRU(64)(embedded)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(encoded)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        x_train,
        train_y,
        validation_data=(x_validation, validation_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    model.save(model_dir / "model.keras")
    (model_dir / "tokenizer.json").write_text(
        tokenizer.to_json(), encoding="utf-8"
    )
    model_config = {
        "model_type": model_name,
        "max_length": args.max_length,
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
    }
    (model_dir / "config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )

    validation_predictions = model.predict(x_validation, batch_size=args.batch_size)
    validation_f1 = float(
        f1_score(
            validation_y,
            validation_predictions.argmax(axis=1),
            average="weighted",
            zero_division=0,
        )
    )
    test_predictions = model.predict(x_test, batch_size=args.batch_size)
    test_metrics = _save_metrics(
        model_name,
        test_y,
        test_predictions.argmax(axis=1),
        class_names,
        output_dir,
    )
    return validation_f1, test_metrics


def _fit_transformer(
    *,
    train_texts: list[str],
    validation_texts: list[str],
    test_texts: list[str],
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    class_names: list[str],
    class_weights: dict[int, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[float, dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_dir = output_dir / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.transformer_name,
        num_labels=len(class_names),
        id2label={index: label for index, label in enumerate(class_names)},
        label2id={label: index for index, label in enumerate(class_names)},
    )

    class ComplaintDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> tuple[str, int]:
            return self.texts[index], int(self.labels[index])

    def collate(
        items: list[tuple[str, int]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        texts, labels = zip(*items)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        return encoded, torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(
        ComplaintDataset(train_texts, train_y),
        batch_size=args.transformer_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        ComplaintDataset(validation_texts, validation_y),
        batch_size=args.transformer_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ComplaintDataset(test_texts, test_y),
        batch_size=args.transformer_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights = torch.tensor(
        [class_weights[index] for index in range(len(class_names))],
        dtype=torch.float,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_validation_f1 = 0.0
    patience_remaining = 2

    def predict(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
        model.eval()
        true_labels: list[int] = []
        predictions: list[int] = []
        total_loss = 0.0
        with torch.no_grad():
            for encoded, labels in loader:
                encoded = {key: value.to(device) for key, value in encoded.items()}
                labels = labels.to(device)
                logits = model(**encoded).logits
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                true_labels.extend(labels.cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        average_loss = total_loss / max(len(true_labels), 1)
        return np.asarray(true_labels), np.asarray(predictions), average_loss

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for encoded, labels in train_loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item() * labels.size(0)

        val_true, val_pred, validation_loss = predict(validation_loader)
        validation_f1 = float(
            f1_score(val_true, val_pred, average="weighted", zero_division=0)
        )
        train_loss = total_train_loss / max(len(train_y), 1)
        print(
            f"transformer epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
            f"val_weighted_f1={validation_f1:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_f1 = validation_f1
            patience_remaining = 2
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            (model_dir / "training_config.json").write_text(
                json.dumps({"max_length": args.max_length}, indent=2),
                encoding="utf-8",
            )
        else:
            patience_remaining -= 1
            if patience_remaining == 0:
                break

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    val_true, val_pred, _ = predict(validation_loader)
    best_validation_f1 = float(
        f1_score(val_true, val_pred, average="weighted", zero_division=0)
    )
    test_true, test_pred, _ = predict(test_loader)
    test_metrics = _save_metrics(
        "transformer", test_true, test_pred, class_names, output_dir
    )
    return best_validation_f1, test_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark complaint classification models."
    )
    parser.add_argument("--data", type=Path, default=Path("complaints_processed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Models to train; defaults to all four.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-vocab-size", type=int, default=50_000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--transformer-name", default="distilbert-base-uncased")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("epochs, batch-size, and max-length must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)

    texts, labels, dropped = load_dataset(args.data)
    (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    ) = stratified_split(texts, labels, random_state=args.seed)
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_labels)
    validation_y = label_encoder.transform(validation_labels)
    test_y = label_encoder.transform(test_labels)
    class_names = label_encoder.classes_.tolist()
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=train_y,
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "usable_examples": len(texts),
                "dropped_empty_narratives": dropped,
                "class_counts": {
                    label: labels.count(label) for label in class_names
                },
                "split_sizes": {
                    "train": len(train_texts),
                    "validation": len(validation_texts),
                    "test": len(test_texts),
                },
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scores: dict[str, dict[str, float]] = {}
    for model_name in args.models:
        print(f"\nTraining {model_name} ...")
        if model_name in RECURRENT_MODELS:
            validation_f1, test_metrics = _fit_recurrent_model(
                model_name,
                train_texts=train_texts,
                validation_texts=validation_texts,
                test_texts=test_texts,
                train_y=train_y,
                validation_y=validation_y,
                test_y=test_y,
                class_names=class_names,
                class_weights=class_weights,
                args=args,
                output_dir=args.output_dir,
            )
        else:
            validation_f1, test_metrics = _fit_transformer(
                train_texts=train_texts,
                validation_texts=validation_texts,
                test_texts=test_texts,
                train_y=train_y,
                validation_y=validation_y,
                test_y=test_y,
                class_names=class_names,
                class_weights=class_weights,
                args=args,
                output_dir=args.output_dir,
            )
        scores[model_name] = {
            "validation_weighted_f1": validation_f1,
            **{f"test_{metric}": value for metric, value in test_metrics.items()},
        }

    best_model = max(scores, key=lambda name: scores[name]["validation_weighted_f1"])
    (args.output_dir / "comparison.json").write_text(
        json.dumps(scores, indent=2), encoding="utf-8"
    )
    (args.output_dir / "best_model.json").write_text(
        json.dumps({"model_type": best_model, "path": best_model}, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "validation_weighted_f1",
                "test_accuracy",
                "test_macro_precision",
                "test_macro_recall",
                "test_macro_f1",
                "test_weighted_precision",
                "test_weighted_recall",
                "test_weighted_f1",
            ],
        )
        writer.writeheader()
        for model_name, model_scores in scores.items():
            writer.writerow({"model": model_name, **model_scores})
    print(f"\nBest model by validation weighted F1: {best_model}")
    print(f"Evaluation outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
