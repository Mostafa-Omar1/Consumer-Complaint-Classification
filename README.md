# Consumer Complaint Classification

An end-to-end text classification project that compares three recurrent neural
networks built with Keras (SimpleRNN, LSTM, and GRU) with a fine-tuned
Hugging Face transformer. It predicts the product category for a customer's
written complaint.

## Pipeline

1. Load a CSV containing `product` and `narrative` columns.
2. Lowercase text, remove punctuation, digits, and special characters, and
   filter English stop words. Negation words such as `no` and `not` are kept.
   Empty narratives after preprocessing are excluded and counted.
3. Make reproducible stratified train/validation/test splits (70%/10%/20%).
4. Train SimpleRNN, LSTM, and GRU models using a training-only vocabulary,
   learned embeddings, padded sequences, class weights, and early stopping.
5. Fine-tune `distilbert-base-uncased` with class-weighted loss and early
   stopping.
6. Report accuracy, macro/weighted precision, recall and F1, plus confusion
   matrices. Select the deployable model by validation weighted F1; test scores
   are kept for final comparison rather than model selection.
7. Classify new complaints with the selected checkpoint.

## Dataset

The expected input is a UTF-8 CSV with the columns `product` and `narrative`.
For example, the local `complaints_processed.csv` works with this project.
The original source URL was not provided with the dataset, so this repository
does not claim provenance or redistribute complaint narratives. Keep the CSV
out of version control; the default `.gitignore` excludes it.

Place your authorized dataset at `complaints_processed.csv`, or pass its path
with `--data`. Before publishing or sharing any dataset, verify its license,
provenance, and privacy requirements.

## Setup

Python 3.10, 3.11, or 3.12 is required by the TensorFlow dependency.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first transformer run downloads the selected pretrained checkpoint from
Hugging Face. A CUDA-capable PyTorch installation is used automatically when
available; otherwise training runs on CPU.

## Train and compare all four models

```bash
python -m complaint_classifier.training --data complaints_processed.csv
```

To train only a subset, for example:

```bash
python -m complaint_classifier.training --data complaints_processed.csv --models rnn lstm gru --epochs 5
```

Useful options include `--max-length`, `--max-vocab-size`, `--batch-size`,
`--transformer-batch-size`, `--transformer-name`, `--seed`, and `--output-dir`.
The default output directory is `outputs/`. It contains per-model metrics and
confusion-matrix PNGs, `comparison.csv`, `comparison.json`, the label mapping,
and `best_model.json`. Model checkpoints are also saved there and are ignored
by Git.

## Classify a new complaint

After training:

```bash
python -m complaint_classifier.predict \
  "I was charged twice for a payment on my credit card."
```

The command prints the predicted category, confidence, model name, and class
probabilities as JSON.

## Tests

```bash
python -m unittest discover -s tests -v
```

The data tests do not require the deep-learning dependencies.
