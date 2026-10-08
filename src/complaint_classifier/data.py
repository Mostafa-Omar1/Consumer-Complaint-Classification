"""Dataset loading, text cleaning, and reproducible stratified splits."""

from __future__ import annotations

import csv
import re
from pathlib import Path


_NON_LETTERS = re.compile(r"[^a-z]+")
_STOP_WORDS = frozenset(
    """
    a an the and but if or because as until while of at by for with about against
    between into through during before after above below to from up down in out
    on off over under again further then once here there when where why how all
    any both each few more most other some such only own same so than too very
    can will just should now i me my myself we our ours ourselves you your yours
    yourself yourselves he him his himself she her hers herself it its itself
    they them their theirs themselves what which who whom this that these those
    am is are was were be been being have has had having do does did doing
    would could ought im ive youre youve hes shes theyre theyve thats whats whos
    whom its were youll well theyll
    """.split()
)


def clean_text(text: str) -> str:
    """Lowercase text, remove digits/punctuation, and filter English stop words."""
    words = _NON_LETTERS.sub(" ", text.lower()).split()
    return " ".join(word for word in words if word not in _STOP_WORDS)


def load_dataset(path: str | Path) -> tuple[list[str], list[str], int]:
    """Load complaint narratives and product labels from a CSV file.

    Returns cleaned texts, labels, and the number of rows excluded because the
    narrative was empty before or after preprocessing.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    texts: list[str] = []
    labels: list[str] = []
    dropped = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        required = {"product", "narrative"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Dataset must contain columns {sorted(required)}; "
                f"missing {sorted(missing)} in {fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            label = (row.get("product") or "").strip()
            narrative = row.get("narrative") or ""
            if not label:
                raise ValueError(f"Missing product label on CSV row {row_number}")
            cleaned = clean_text(narrative)
            if not cleaned:
                dropped += 1
                continue
            texts.append(cleaned)
            labels.append(label)

    if not texts:
        raise ValueError(f"No usable complaint narratives found in {csv_path}")
    return texts, labels, dropped


def stratified_split(
    texts: list[str],
    labels: list[str],
    *,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Create stratified train/validation/test splits in a 70/10/20 ratio."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must contain the same number of rows")
    if not texts:
        raise ValueError("Cannot split an empty dataset")

    from sklearn.model_selection import train_test_split

    class_counts: dict[str, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    small_classes = {label: count for label, count in class_counts.items() if count < 3}
    if small_classes:
        raise ValueError(
            "Each class needs at least 3 examples for stratified splitting; "
            f"found {small_classes}"
        )

    train_texts, remaining_texts, train_labels, remaining_labels = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    validation_texts, test_texts, validation_labels, test_labels = train_test_split(
        remaining_texts,
        remaining_labels,
        test_size=2 / 3,
        random_state=random_state,
        stratify=remaining_labels,
    )
    return (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    )
