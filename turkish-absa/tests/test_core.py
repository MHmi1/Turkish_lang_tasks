"""Network-free checks of data quality and supervised label handling."""
import unittest

from absa.data import (build_implicit_ontology, canonical_aspect, encode_apc,
                       locate_spans, safe_negative_candidates)
from absa.infer import ATEPredictor
from absa.metrics import apc_trainer_metrics
import numpy as np
from absa.quality import canonical_text, clean_training, filter_near_heldout, quality_flags
from absa.text import tr_lower
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

    def test_partial_alignment_rejects_whole_document(self):
        issues = []
        found = locate_spans("Kargo hızlı", [
            {"span": "Kargo", "aspect": "teslimat", "polarity": "positive"},
            {"span": "batarya", "aspect": "batarya", "polarity": "negative"}], issues)
        self.assertEqual(found, [])
        self.assertEqual(len(issues), 1)

    def test_unsupported_polarity_rejects_document(self):
        issues = []
        self.assertEqual(locate_spans("Kargo hızlı", [
            {"span": "Kargo", "aspect": "teslimat", "polarity": "conflict"}], issues), [])
        self.assertEqual(issues[0]["reason"], "unsupported_polarity")

    def test_threshold_controls_ate_predictions(self):
        offsets = np.array([[0, 5]])
        probs = np.array([[0.51, 0.48, 0.01]])
        self.assertEqual(ATEPredictor._decode_one("Kargo", offsets, probs, 0.50), [])
        self.assertEqual(ATEPredictor._decode_one("Kargo", offsets, probs, 0.45)[0]["span"], "Kargo")

    def test_domain_ontology_and_token_bounded_cues(self):
        doc = row()
        doc["domain"] = "shopping"
        ontology = build_implicit_ontology([doc, doc])
        self.assertEqual(ontology, {"shopping": ["teslimat"]})
        self.assertIn("ses", safe_negative_candidates("Basit ürün", [], ["ses"]))
        self.assertNotIn("ses", safe_negative_candidates("bas çok güçlü", [], ["ses"]))

    def test_turkish_case_mapping(self):
        self.assertEqual(tr_lower("IŞIK İYİ"), "ışık iyi")

    def test_near_duplicate_heldout_removed(self):
        held = row("Kargo hızlı geldi, paket temiz ve sağlamdı. Ürün çalışıyor.", "held")
        train = row("Kargo hızlı geldi, paket temiz ve sağlamdı! Ürün çalışıyor.", "train")
        kept, report = filter_near_heldout([train], {"test": [held]}, threshold=0.85)
        self.assertEqual(kept, [])
        self.assertEqual(report["removed"], 1)

    def test_apc_validation_aggregates_windows_per_gold_pair(self):
        logits = np.array([[0, 0, 3, 0], [0, 0, 1, 0], [0, 0, 0, 3]])
        labels = np.array([2, 2, 3])
        metadata = [{"pair_id": 0}, {"pair_id": 0},
                    {"pair_id": 1, "negative_sample": True}]
        metrics = apc_trainer_metrics((logits, labels), metadata)
        self.assertEqual(metrics["gold_pairs"], 1)
        self.assertEqual(metrics["accuracy"], 1)

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
