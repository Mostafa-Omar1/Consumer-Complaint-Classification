import csv
import tempfile
import unittest
from pathlib import Path

from complaint_classifier.data import clean_text, load_dataset, stratified_split


class CleanTextTests(unittest.TestCase):
    def test_removes_punctuation_numbers_and_stop_words(self) -> None:
        self.assertEqual(
            clean_text("I did NOT receive $100.00, in 2024!"),
            "not receive",
        )

    def test_keeps_negation_words(self) -> None:
        self.assertEqual(clean_text("No payment was never made."), "no payment never made")


class LoadDatasetTests(unittest.TestCase):
    def test_loads_rows_and_counts_empty_narratives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complaints.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["product", "narrative"])
                writer.writeheader()
                writer.writerow({"product": "credit_card", "narrative": "Wrong charge 42"})
                writer.writerow({"product": "debt_collection", "narrative": "123 !!!"})

            texts, labels, dropped = load_dataset(path)

        self.assertEqual(texts, ["wrong charge"])
        self.assertEqual(labels, ["credit_card"])
        self.assertEqual(dropped, 1)

    def test_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("category,text\nx,y\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain columns"):
                load_dataset(path)


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_preserves_each_class(self) -> None:
        texts = [f"complaint {index}" for index in range(100)]
        labels = ["credit_card"] * 50 + ["debt_collection"] * 50

        first = stratified_split(texts, labels, random_state=42)
        second = stratified_split(texts, labels, random_state=42)

        self.assertEqual(first, second)
        train_texts, validation_texts, test_texts, train_y, validation_y, test_y = first
        self.assertEqual(
            (len(train_texts), len(validation_texts), len(test_texts)),
            (70, 10, 20),
        )
        for split_labels in (train_y, validation_y, test_y):
            self.assertEqual(set(split_labels), set(labels))

    def test_rejects_mismatched_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            stratified_split(["complaint"], ["credit_card", "debt_collection"])


if __name__ == "__main__":
    unittest.main()
