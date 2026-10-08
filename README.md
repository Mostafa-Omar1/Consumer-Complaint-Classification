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

The dataset summary includes per-class counts, percentages, the largest-to-
smallest class ratio, and split sizes.

## Benchmark results

Measured on the held-out test split of the local dataset after three training
epochs (seed 42, maximum sequence length 256). The stratified split contained
113,687 training, 16,241 validation, and 32,483 test examples. The checkpoint
was selected using validation weighted F1.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| SimpleRNN | 73.27% | 66.32% | 70.87% | 66.81% | 73.99% |
| LSTM | 82.55% | 75.82% | 83.84% | 79.21% | 83.04% |
| GRU | 85.26% | 79.62% | 85.38% | 82.09% | 85.62% |
| DistilBERT | **88.05%** | **84.27%** | 85.39% | **84.80%** | **88.08%** |

DistilBERT was the best model on validation weighted F1 and also achieved the
highest held-out test accuracy and weighted F1. The dataset CSV and model
checkpoints are intentionally excluded from the public repository; metrics
reflect the local CSV used for this run.

## Notebook walkthrough

Open [Consumer_Complaint_Classification.ipynb](./Consumer_Complaint_Classification.ipynb)
in VS Code, select the project virtual environment as the notebook kernel, and run
the cells from top to bottom. Create the environment with the setup commands below;
the notebook requires Python 3.10–3.12. The notebook explores class balance, streams
model-training progress, compares the test metrics and confusion matrices, and
classifies an illustrative new complaint.

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
python -m pip install -e ".[notebook]"
```

The first transformer run downloads the selected pretrained checkpoint from
Hugging Face. A CUDA-capable PyTorch installation is used automatically when
available; otherwise training runs on CPU.
On Windows with Python 3.11 or 3.12, the `msvc-runtime` dependency provides
the Microsoft C++ runtime DLLs inside the Python environment, so a
machine-wide administrator install is not required. Python 3.10 users may
need to install the Microsoft Visual C++ Redistributable separately.
The default PyTorch package may be CPU-only. For an NVIDIA GPU, install a
CUDA-enabled PyTorch wheel compatible with your installed driver using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). For
example, the `torch==2.5.1` CUDA 12.1 wheel was verified on an NVIDIA driver
that supports CUDA 12.1:

```bash
python -m pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

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
