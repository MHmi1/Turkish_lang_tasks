import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from absa.data import locate_spans, encode_apc, canonical_aspect, write_jsonl, read_jsonl
from absa.hard_cases import curated_stress_set
from absa.synth_gemini import parse_sample, _ModelRotator, _extract_samples_from_text
from absa.synth_local import generate
from run_all import _gold_apc_pairs


class FakeTokenizer:
    def __call__(self, texts, pairs, **kwargs):
        assert len(texts) == len(pairs)
        return {"input_ids": [[1,2,3] for _ in texts],
                "attention_mask": [[1,1,1] for _ in texts]}


class OfflineTests(unittest.TestCase):
    def test_marked_gemini_parse_offsets(self):
        sample = {"marked_text":"abi [[ASP|kamerası|positive]] çok iyi ama [[ASP|batarya|negative]] çöp",
                  "register":"gen_z","implicit_aspects":[{"aspect":"teslimat","polarity":"neutral"}]}
        o = parse_sample(sample, "gen_z_social_slang", "electronics")
        self.assertIsNotNone(o)
        self.assertEqual(len(o['aspects']),3)
        for a in o['aspects']:
            if a['span'] is not None:
                self.assertEqual(o['text'][a['start']:a['end']],a['span'])
        self.assertIn('gen_z',o['flags'])

    def test_invalid_marker_rejected(self):
        self.assertIsNone(parse_sample({"marked_text":"ürün [[ASP|kargo|good]] berbat", "register":"informal"},"heavy_sarcasm_irony","ecommerce"))

    def test_no_fabricated_implicit_on_unmatched_explicit(self):
        self.assertEqual(locate_spans('Ürün kırık geldi',[{'span':'bataryası','aspect':'batarya','polarity':'negative'}]),[])

    def test_curated_stress_spans(self):
        docs=curated_stress_set()
        self.assertGreaterEqual(len(docs),30)
        self.assertGreater(sum(1 for d in docs if any(a['span'] is None for a in d['aspects'])),8)
        for d in docs:
            for a in d['aspects']:
                if a['span'] is not None:
                    self.assertEqual(d['text'][a['start']:a['end']],a['span'])

    def test_generator_sanity(self):
        docs=generate(n=120,seed=1)
        self.assertEqual(len(docs),120)
        for d in docs:
            self.assertTrue(d['aspects'])
            for a in d['aspects']:
                if a['span'] is not None:
                    self.assertEqual(d['text'][a['start']:a['end']],a['span'])

    def test_apc_four_class_and_absent_examples(self):
        doc={'id':'x','text':'Kargo çok hızlı, ürün güzel.','flags':[],
             'aspects':[{'span':'Kargo','aspect':'kargo','polarity':'positive','start':0,'end':5}]}
        items,meta=encode_apc(FakeTokenizer(),[doc],train=True,negative_per_doc=3)
        labels=[x['labels'] for x in items]
        self.assertIn(3,labels)
        self.assertIn(2,labels)
        self.assertEqual(len(meta),len(items))

    def test_conflicting_apc_targets_are_excluded(self):
        doc = {'id':'conflict','text':'Pil önce iyiydi ama artık kötü.','flags':[],
               'aspects':[{'span':'Pil','aspect':'batarya','polarity':'positive','start':0,'end':3},
                          {'span':'Pil','aspect':'batarya','polarity':'negative','start':0,'end':3}]}
        items, meta = encode_apc(FakeTokenizer(), [doc], train=True, negative_per_doc=0)
        self.assertEqual(items, [])
        self.assertEqual(meta, [])
        self.assertEqual(_gold_apc_pairs([doc])[2], [])

    def test_target_aliases(self):
        self.assertEqual(canonical_aspect('PİL ÖMRÜ'), 'batarya')
        self.assertEqual(canonical_aspect('fiyatı'),'fiyat')

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data.jsonl'
            write_jsonl(p,curated_stress_set()[:2])
            self.assertEqual(len(read_jsonl(p)),2)

    def test_gemini_model_rotator_fallback(self):
        # Keep the smoke suite genuinely offline and deterministic.
        with patch('absa.synth_gemini.requests.get', side_effect=OSError('offline')):
            rot = _ModelRotator("fake-key", preferred="gemini-3.8-flash,gemini-3.5-flash-lite", min_interval=0.0)
        self.assertGreaterEqual(len(rot.models), 4)
        m1, _, _ = rot.acquire()
        self.assertEqual(m1, "gemini-3.8-flash")
        rot.mark_rate_limited("gemini-3.8-flash", "Quota exceeded for quota metric 'Requests per day'")
        self.assertIn("gemini-3.8-flash", rot.dead)
        m2, _, _ = rot.acquire()
        self.assertNotEqual(m2, "gemini-3.8-flash")
        parsed = _extract_samples_from_text("```json\n{\"samples\": [{\"marked_text\": \"x\", \"register\": \"gen_z\"}]}\n```")
        self.assertEqual(len(parsed), 1)


if __name__ == '__main__':
    unittest.main()
