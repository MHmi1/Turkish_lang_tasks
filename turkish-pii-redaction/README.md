# Turkish PII detection and redaction

A hybrid Turkish personal information detector. A ModernBERT TR token classifier proposes spans, deterministic checks validate structured identifiers, and an inference layer resolves overlaps before redaction. The code supports 13 entity classes and includes training, evaluation, a FastAPI service, and a preserved Kaggle execution record.

![PII pipeline](assets/pipeline.svg)

## Reproduce the pipeline

```bash
python -m pip install -r requirements.txt
python run_all.py --stage all --no_gemini
```

The full run downloads public datasets and a pretrained encoder, trains the `clean` and `full` variants, calibrates a decision threshold on validation data, evaluates test sets, and exports the selected model. `clean` uses public corpora; `full` adds synthetic edge cases. The bundled `gemini_raw.jsonl` is a synthetic seed from the recorded run and enables a run without live Gemini requests. The recorded run generated additional Gemini examples, so using only the seed reproduces the pipeline but **not the exact historical training set or scores**. New generation is optional: supply `GEMINI_API_KEY` through the environment. No credential is stored here. `--quick` is a pipeline check, not a benchmark.

The historical run is preserved in [the executed Kaggle notebook](notebooks/kaggle_execution.ipynb) and [its text log](artifacts/kaggle_run.log). The notebook launch cell contains the original Kaggle input path. It documents evaluation and export, but **does not document training**: the recorded `train` stage reports that both checkpoints already existed and ran zero training jobs. The source files here are the cleaned public copy. Weights, training curves, per document predictions, and plot images from the earlier training run are **not included**. Scores below can be checked against the log but cannot be independently recomputed from this repository without rerunning.

## Architecture

The model uses [ModernBERT TR base](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base) with a 13 class BIO taxonomy: person, address, phone, email, date, URL, account number, secret, TCKN, VKN, IBAN, card, and plate. `pii/rules.py` checks Turkish identity and tax numbers, IBANs, payment cards, phone numbers, and emails. `pii/infer.py` uses windowed inference for long documents, handles Turkish case suffixes, merges neural and validated rule spans, and applies redaction. Evaluation reports character overlap F2, exact span F1, leak rate, missed spans, and false redaction where available.

## Recorded results

These are **shared-label overlap F2 scores from one Kaggle run**, not a ranking against published papers. Each row names a **test dataset**, not a different earlier model; the only model baseline is the [YTU PII checkpoint](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-pii-ner). It used a fixed 0.5 threshold and neural predictions alone; `full+hybrid` used a validation-calibrated 0.6 threshold plus structural regex rules. Both went through this project's decoder and metric implementation, but this is **not an equal post-processing comparison**. The YTU model card demonstrates wordpiece aggregation with `aggregation_strategy="first"`; this benchmark instead decodes every subword for both models. The effect on the YTU scores has not been measured. Intervals are separate document bootstrap 95% intervals, not a paired confidence interval for the difference.

| Dataset | Documents | YTU F2 [95% CI] | Full + hybrid F2 [95% CI] |
|---|---:|---:|---:|
| [NewMind KVKK **test_hard**](https://huggingface.co/datasets/newmindai/nm-kvkk-pii-6K) | **241** | 0.922 [0.913, 0.930] | 0.924 [0.908, 0.939] |
| [Fevzi corpus](https://huggingface.co/datasets/fevziegeyurtsevenler/turkish-pii-corpus) | 62 | 0.708 [0.621, 0.782] | 0.960 [0.926, 0.987] |
| [Belgin](https://huggingface.co/datasets/negentropi/belgin-pii-dataset) | 200 | 0.711 [0.664, 0.762] | 0.986 [0.976, 0.995] |
| [BTX24](https://huggingface.co/datasets/BTX24/turkish-privacy-pii-ner) | 10,393 | 0.557 [0.549, 0.565] | 0.986 [0.984, 0.989] |
| [Yusuf privacy filter](https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset) | 405 | 0.548 [0.521, 0.573] | 0.997 [0.993, 1.000] |

The NewMind dataset has 602 documents in its **whole test** split, but this code loads only `spans/test_hard.jsonl`, which has **241**. Its 118-label ontology is much wider than ours. In the currently published hard split, there are 1,788 gold mentions; this evaluation scores only 1,151 mapped mentions (64.4%), including 848 person-name mentions. Unmapped gold regions are ignored, and predictions overlapping them are removed before scoring. Thus 0.924 is a **restricted-label result on synthetic multi-person documents**, not full-ontology PII performance. The 602-document count in the first README release was incorrect.

On this restricted NewMind view, the project also recorded **0.828 exact span F1**, **0.0044 uncovered-gold-character rate**, and **0.0070 completely missed-span rate**. The YTU baseline exact F1 was 0.534 under this decoder. Exact boundaries differ greatly, but overlap F2 differs by only 0.002. The overlap metric counts any positive intersection between same-label spans as a match, even if their boundaries differ. No paired significance test was run. The character metric measures coverage regardless of the predicted entity type; it should not be read as a complete privacy-risk estimate. The hard split has no fully negative documents, so it does not measure document-level false redaction.

The large gaps in the other rows are diagnostic, not evidence of a general 25–45 percentage point advance. BTX24 is entirely synthetic and 73.35% of its test rows have a masked template skeleton also present in training. Belgin and Yusuf test rows come from sources also used for training. Fevzi is only 62 test rows and **regex alone scores 0.959 F2**, nearly matching the 0.960 hybrid result. The `clean` variant scored 1.000 F2 on BTX24 and Belgin, so the additional synthetic data did not create those near-perfect results.

Synthetic stress results are useful for failure analysis, not independent validation: `full+hybrid` reached 0.870 F2 on Gemini generated edge cases and 0.990 on locally generated cases. The Gemini stress set had a **0.433 false redaction rate** for the project. This is a substantial precision risk before use with real user text.

## Relationship to prior work

The [YTU PII model](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-pii-ner) is a recent ModernBERT TR system, not an old method. Its card reports 25 entity types, 7,763 retained training documents from real-document and synthetic sources, and 0.8101 overlap micro-F2 on **its own** 300-document test. This project's preparation stage reported 67,903 public training examples before optional synthetic augmentation and a narrower 13-class taxonomy; the archived session does not show the earlier training jobs. The YTU card's 0.8101 cannot be compared directly with our table because the test set and label policy differ. The [ModernBERT paper](https://aclanthology.org/2025.acl-long.127/) motivates both encoders. The [Horizon Labs model card](https://huggingface.co/Horizon-Labs/pii-redactor-base) reports a Turkish redaction F1 of 0.851 under another policy and metric; that score cannot be ranked against our overlap F2 either.

The evidence supports a working pipeline, not a state-of-the-art or no-overfitting claim. The exported `full` variant was chosen using public and synthetic validation scores, so its synthetic stress result is especially unsuitable as an independent gain claim. A fairer comparison needs the YTU checkpoint calibrated on the same validation set, **both models with and without the same rule layer**, a paired document bootstrap, and scores on unseen templates and all relevant labels. Rerun weights, per-document predictions, package and dataset revisions, and an independently annotated Turkish test set are still needed.

## Files

`pii/` contains data preparation, synthetic generation, model training, inference, rules, metrics, and plotting. `run_all.py` orchestrates stages, `kaggle_run.py` launches on Kaggle, and `serve.py` serves redaction. `notebooks/` and `artifacts/` hold the execution evidence.
