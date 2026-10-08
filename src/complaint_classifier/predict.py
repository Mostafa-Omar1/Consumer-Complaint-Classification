"""Classify a new complaint with the model selected during training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from complaint_classifier.data import clean_text


def predict_complaint(text: str, output_dir: str | Path = "outputs") -> dict[str, object]:
    root = Path(output_dir)
    best_model = json.loads((root / "best_model.json").read_text(encoding="utf-8"))
    model_type = best_model["model_type"]
    model_dir = root / best_model["path"]
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Complaint has no words left after preprocessing")

    if model_type == "transformer":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = json.loads(
            (model_dir / "training_config.json").read_text(encoding="utf-8")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        encoded = tokenizer(
            cleaned,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits[0], dim=0).tolist()
    else:
        import tensorflow as tf

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            (model_dir / "tokenizer.json").read_text(encoding="utf-8")
        )
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=config["max_length"],
            padding="post",
            truncating="post",
        )
        model = tf.keras.models.load_model(model_dir / "model.keras")
        probabilities = model.predict(padded, verbose=0)[0].tolist()

    class_names = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    predicted_index = int(np.argmax(probabilities))
    return {
        "category": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "model": model_type,
        "probabilities": {
            label: float(probability)
            for label, probability in zip(class_names, probabilities)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a customer complaint.")
    parser.add_argument("complaint", help="Complaint text to classify")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(json.dumps(predict_complaint(args.complaint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
