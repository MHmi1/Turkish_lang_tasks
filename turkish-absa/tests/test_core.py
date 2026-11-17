"""Network-free checks of data quality and supervised label handling."""
import unittest

from absa.data import canonical_aspect, encode_apc, locate_spans
from absa.quality import canonical_text, clean_training, quality_flags
from run_all import _gold_apc_pairs


class FakeTokenizer:
    def __call__(self, texts, pairs, **kwargs):
        return {"input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts]}


def row(text="Kargo hızlı.", row_id="one"):
    return {"id": row_id, "text": text, "source": "public_fixture", "flags": [],
            "aspects": [{"span": text[:5], "start": 0, "end": 5,
                         "aspect": "kargo", "polarity": "positive"}]}


class CoreTests(unittest.TestCase):
    def test_normalized_overlap_removed_without_text_rewrite(self):
        train = [row(), row("Kargo  hızlı.", "two")]
        cleaned, report = clean_training(train, {"val": [row("Kargo hızlı.", "val")]})
        self.assertEqual(cleaned, [])
        self.assertEqual(report["dropped_by_reason"]["normalized_heldout_overlap"], 2)

    def test_valid_offset_preserved(self):
        doc = row("Kargo çok hızlı.")
        cleaned, report = clean_training([doc], {"val": []})
        self.assertEqual(cleaned[0]["text"][0:5], "Kargo")
        self.assertEqual(report["counts"]["accepted"], 1)

    def test_bad_offset_rejected(self):
        doc = row()
        doc["aspects"][0]["end"] = 4
        cleaned, report = clean_training([doc], {})
        self.assertFalse(cleaned)
        self.assertEqual(report["dropped_by_reason"]["invalid_span"], 1)

    def test_diagnostic_regex_does_not_mutate(self):
        text = "Kargo çoook iyi!!! www.example.org"
        self.assertIn("url", quality_flags(text))
        self.assertEqual(canonical_text(text), text.casefold())

    def test_unaligned_explicit_label_not_fabricated(self):
        self.assertEqual(locate_spans("Kargo hızlı", [{"span": "batarya", "aspect": "batarya",
                                                       "polarity": "negative"}]), [])

    def test_ambiguous_apc_pair_excluded(self):
        doc = row()
        doc["aspects"].append({**doc["aspects"][0], "polarity": "negative"})
        items, _ = encode_apc(FakeTokenizer(), [doc], train=True, negative_per_doc=0)
        self.assertEqual(items, [])
        self.assertEqual(_gold_apc_pairs([doc])[2], [])

    def test_aliases(self):
        self.assertEqual(canonical_aspect("fiyatı"), "fiyat")


if __name__ == "__main__":
    unittest.main()
