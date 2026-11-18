"""Build an English, public-data-only, source-visible Colab notebook."""
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Turkish_ABSA_Colab.ipynb"
CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": source.splitlines(keepends=True)})


def public_runner_source():
    """Use the shared implementation while omitting optional synthetic stages."""
    source = (ROOT / "run_all.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = ("stage_prepare", "stage_clean", "stage_eda", "stage_train",
             "_gold_apc_pairs", "stage_eval", "format_markdown", "stage_export")
    functions = []
    for name in names:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        segment = ast.get_source_segment(source, node)
        functions.append(segment)
    prelude = '''"""Public-data-only Colab runner, generated from run_all.py."""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter

from absa import config as C
from absa.data import load_public, read_jsonl, write_jsonl

S = C.SPLITS
'''
    return prelude + "\n\n" + "\n\n".join(functions) + "\n"


def public_config_source():
    """Exclude CLI-only generation settings without editing the shared config."""
    source = (ROOT / "absa/config.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    omitted = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"GEMINI_MODEL", "GEMINI_MODEL_POOL"}
                for target in node.targets):
            omitted.update(range(node.lineno, node.end_lineno + 1))
    result = "".join(line for number, line in enumerate(lines, 1)
                     if number not in omitted and not line.startswith("# Candidate models for the optional synthetic"))
    result = result.replace('SYNTH = DATA / "synth"\n', '')
    result = result.replace('for _p in (SPLITS, SYNTH, CACHE,', 'for _p in (SPLITS, CACHE,')
    if "SYNTH" in result or "GEMINI" in result:
        raise RuntimeError("Public notebook config still contains generation settings")
    return result


md("""# Turkish ABSA: public-data Colab experiment

This notebook trains a two-stage Turkish aspect-based sentiment analysis system: **aspect term extraction (ATE)** and **aspect polarity classification (APC)**. It uses public ABSA datasets, a conservative quality gate, a Turkish ModernBERT encoder, held-out evaluation, and error analysis. All executable project source is visible below; no repository upload is required.

**Status:** No new score is claimed until this notebook has run successfully. Use a GPU runtime, run cells in order, and keep the generated report bundle. `QUICK_MODE=True` checks the pipeline only and must not be reported as a benchmark.

**Data:** [ABSA-TR](https://huggingface.co/datasets/ytu-ce-cosmos/absa-tr) supplies Turkish span and polarity labels; [M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA) supplies Turkish data across seven domains; [Clothing Products ABSA](https://huggingface.co/datasets/ebrukilic/clothing_products_ABSA) supplies category-level polarity without spans. The model starts from [ModernBERT-TR](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base). Published [ATE](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-absa-extraction) and [APC](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-absa-polarity) checkpoints are optional comparison baselines.

ABSA-TR annotations include model-generated labels; M-ABSA is translated and reviewed; clothing labels have a different supervision level. Scores are therefore reported **per source**. The training split is cleaned, while validation and test annotations stay untouched. No generated reviews are used in this notebook.

The public YTU ATE/APC checkpoints are relevant recent baselines, but their training mixtures and preprocessing differ from this experiment. Their published scores are not directly comparable with a new run here. The notebook validates checkpoint label maps and reports only same-code, same-split comparisons.

Dataset selection was checked against newer resources: [DimABSA 2026](https://aclanthology.org/2026.acl-long.1881.pdf) does not include Turkish and uses dimensional sentiment labels. The [201k-row Turkish ABSA upload](https://huggingface.co/datasets/STNM-NLPhoenix/turkish-absa) exposes a single training split and entity-level labels without character offsets in its public viewer. It is left out of this fixed, span-evaluated benchmark pending a separate annotation and split audit.
""")

md("""## 1. Runtime and experiment configuration

Select **Runtime → Change runtime type → GPU**. The notebook stores files under `/content/absa_work`. Set `QUICK_MODE=True` only to check execution. For a portfolio result, run the full configuration and report hardware, dataset counts, seed, package versions, and all held-out metrics.
""")
code("""from pathlib import Path
from types import SimpleNamespace
import json, os, platform, subprocess, sys

PROJECT = Path('/content/turkish_absa_colab')
WORK = Path('/content/absa_work')
for folder in ('absa', 'scripts'):
    (PROJECT / folder).mkdir(parents=True, exist_ok=True)
(PROJECT / 'scripts' / '__init__.py').touch()
sys.path.insert(0, str(PROJECT))
os.environ['ABSA_WORK'] = str(WORK)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

QUICK_MODE = False
SEED = 42
EPOCHS = 5
BATCH_SIZE = 8
GRAD_ACCUM = 4
MAX_LEN = 256
RUN_BASELINE = True
EVAL_IMPLICIT_LIMIT = 120
print('Python:', sys.version.split()[0], '| platform:', platform.platform())
""")

md("""## 2. Project source

These cells write the exact source used by the experiment. The Colab runner is derived from the repository runner and contains only public-data stages. Source files remain editable inside the runtime.
""")
files = [
    "requirements.txt", "absa/__init__.py", "absa/config.py", "absa/text.py", "absa/data.py",
    "absa/quality.py", "absa/metrics.py", "absa/train.py", "absa/infer.py",
    "scripts/audit_data.py", "scripts/eda_data.py", "serve.py",
]
for rel in files:
    src = public_config_source() if rel == "absa/config.py" else (ROOT / rel).read_text(encoding="utf-8")
    code("%%writefile /content/turkish_absa_colab/" + rel + "\n" + src.rstrip("\n") + "\n")
code("%%writefile /content/turkish_absa_colab/run_public.py\n" + public_runner_source())

md("""## 3. Install dependencies and record the environment

The GPU is required for model training. The dependency list and an installed-package snapshot are saved with the output so the result can be reviewed later.
""")
code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT / 'requirements.txt')], check=True)
import torch, transformers, huggingface_hub
print('torch:', torch.__version__, '| transformers:', transformers.__version__, '| hub:', huggingface_hub.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
(WORK / 'reports').mkdir(parents=True, exist_ok=True)
with (WORK / 'reports' / 'environment.txt').open('w') as output:
    output.write(f'Python: {sys.version}\\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}\\n')
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True)
""")

code("""from run_public import stage_prepare, stage_clean, stage_eda, stage_train, stage_eval, stage_export
settings = SimpleNamespace(seed=SEED, quick=QUICK_MODE, no_clothing=False,
                           max_len=MAX_LEN, eda_tokenizer=True, epochs=EPOCHS,
                           bs=BATCH_SIZE, grad_accum=GRAD_ACCUM, lr=3.5e-5,
                           train_real_baseline=False, retrain=False,
                           skip_baseline=not RUN_BASELINE,
                           eval_implicit_limit=EVAL_IMPLICIT_LIMIT)
def run_stage(stage):
    print(f'===== {stage} =====', flush=True)
    globals()[f'stage_{stage}'](settings)
""")

md("""## 4. Download public data and record upstream revisions

The original split assignments are retained. The clothing dataset is used for APC only because it has no span annotations. Dataset revisions are recorded at run time. Internet access is needed for this step and the model downloads.
""")
code("""from huggingface_hub import HfApi
api = HfApi()
repositories = [('dataset', 'ytu-ce-cosmos/absa-tr'),
                ('dataset', 'Multilingual-NLP/M-ABSA'),
                ('dataset', 'ebrukilic/clothing_products_ABSA'),
                ('model', 'ytu-ce-cosmos/modernbert-tr-base'),
                ('model', 'ytu-ce-cosmos/modernbert-tr-absa-extraction'),
                ('model', 'ytu-ce-cosmos/modernbert-tr-absa-polarity')]
revisions = {}
for kind, repo in repositories:
    info = api.dataset_info(repo) if kind == 'dataset' else api.model_info(repo)
    revisions[repo] = info.sha
(WORK / 'reports' / 'upstream_revisions.json').write_text(json.dumps(revisions, indent=2))
os.environ['ABSA_DATASET_REVISIONS'] = json.dumps({repo: revisions[repo] for kind, repo in repositories if kind == 'dataset'})
os.environ['ABSA_MODEL_REVISIONS'] = json.dumps({repo: revisions[repo] for kind, repo in repositories if kind == 'model'})
print(json.dumps(revisions, indent=2))
run_stage('prepare')
manifest = json.loads((WORK / 'data' / 'splits' / 'manifest.json').read_text())
print(json.dumps(manifest, ensure_ascii=False, indent=2))
""")

md("""## 5. Audit, conservative cleaning, and exploratory analysis

The audit checks span offsets, conflicting labels, and split overlap. Cleaning affects **training only**: it removes invalid spans, non-linguistic placeholders, normalized duplicates, normalized text collisions and conservative character five-gram near-duplicates against held-out data. The near-duplicate search is approximate. Documents with any unaligned explicit aspect are excluded as a whole, with counts recorded in the manifest. Regex diagnostics flag URLs, email, phone-like strings, markup and HTML entities, zero-width characters, repeated punctuation, elongated words, low letter ratios, and suspiciously long or short reviews. Flagged text stays in the data unless a separate, explicit rejection rule applies; changing review characters would break gold offsets.

EDA reports source/domain balance, polarity, aspect frequency, explicit versus implicit supervision, multi-aspect reviews, length quantiles, duplicate counts, ambiguous category polarity, and ModernBERT truncation risk. Inspect these reports before training.
""")
code("""subprocess.run([sys.executable, str(PROJECT / 'scripts' / 'audit_data.py'),
                '--work', str(WORK)], cwd=PROJECT, check=True)
audit = json.loads((WORK / 'reports' / 'data_audit.json').read_text())
print('Invalid gold spans:', len(audit['invalid_spans']))
print('Conflicting annotations flagged for review:', len(audit['conflicts']))
if audit['invalid_spans']:
    raise RuntimeError('Invalid gold offsets detected. Inspect data_audit.json before training.')
run_stage('clean')
run_stage('eda')
cleaning = json.loads((WORK / 'reports' / 'cleaning_report.json').read_text())
eda = json.loads((WORK / 'reports' / 'eda_report.json').read_text())
print('Cleaning counts:', cleaning['counts'])
print('Removal reasons:', cleaning['dropped_by_reason'])
print('Token profile:', eda['token_lengths_train_clean_plus_val'])
if eda['token_lengths_train_clean_plus_val']['status'] != 'ok':
    raise RuntimeError('Exact tokenizer EDA failed; inspect eda_report.json.')
if any(row['normalized_text_collisions'] for row in eda['normalized_cross_split_overlap']
       if row['train'] == 'train_clean'):
    raise RuntimeError('Cleaned training data still overlaps a held-out split.')
""")

code("""import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

reports = WORK / 'reports'
summary = pd.read_csv(reports / 'eda_summary.csv')
sources = pd.read_csv(reports / 'eda_source.csv')
polarity = pd.read_csv(reports / 'eda_polarity.csv')
domains = pd.read_csv(reports / 'eda_domain.csv')
noise = pd.read_csv(reports / 'eda_noise.csv')
display(summary)
display(sources.pivot_table(index='split', columns='source', values='documents', fill_value=0))
display(domains[domains['split'] == 'train_clean'].sort_values('documents', ascending=False).head(20))
display(noise[noise['split'] == 'train_public'].sort_values('documents', ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sources[sources['split'] == 'train_clean'].set_index('source')['documents'].sort_values().plot.barh(ax=axes[0], title='Clean training documents by source')
polarity[polarity['split'] == 'train_clean'].set_index('polarity')['aspects'].plot.bar(ax=axes[1], title='Clean training aspect polarity')
for ax in axes: ax.set_xlabel('Count')
fig.tight_layout()
fig.savefig(reports / 'eda_overview.png', dpi=160, bbox_inches='tight')
plt.show()
""")

md("""## 6. Train ATE and APC

ATE learns BIO tags from span-supervised documents. APC learns sentiment for a `(review, aspect)` pair with a distinct `not_mentioned` class; this prevents treating an absent aspect as neutral. Ambiguous same-target polarity views are excluded from APC training. The candidate ontology is learned from cleaned training data by domain. Long reviews use overlapping token windows. APC checkpoint selection uses validation macro F1 on gold pairs under the same target policy as final testing; heuristic absent-aspect negatives remain in validation but are reported separately. Models are trained sequentially on one GPU or in parallel when two GPUs are available. Test splits are reserved for final evaluation.
""")
code("""if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU runtime before training.')
run_stage('train')
for task in ('ate', 'apc'):
    path = WORK / 'models' / f'{task}_full' / 'train_summary.json'
    print(task.upper(), json.loads(path.read_text())['val'])
""")

md("""## 7. Validation error analysis

Use validation errors to diagnose target normalization, negation, and domain mismatch. High-confidence errors are candidates for manual review. This cell does not modify labels or tune on the test sets.
""")
code("""from absa.data import read_jsonl
from absa.infer import APCPredictor
from absa import config as C
from run_public import _gold_apc_pairs
from sklearn.metrics import confusion_matrix

validation = read_jsonl(WORK / 'data' / 'splits' / 'val.jsonl')
texts, targets, gold_ids, metadata = _gold_apc_pairs(validation)
apc = APCPredictor(str(WORK / 'models' / 'apc_full' / 'final'))
pred_ids, confidence, _ = apc.classify_pairs(texts, targets)
gold_names = [C.APC_I2L[i] for i in gold_ids]
pred_names = [apc.label_names[i] for i in pred_ids]
labels = ['negative', 'neutral', 'positive', 'not_mentioned']
display(pd.DataFrame(confusion_matrix(gold_names, pred_names, labels=labels),
                     index=[f'gold:{x}' for x in labels],
                     columns=[f'pred:{x}' for x in labels]))
errors = pd.DataFrame([{'source_id': m['id'], 'text': t, 'target': a,
                        'gold': g, 'prediction': p, 'confidence': round(c, 3)}
                       for t, a, g, p, c, m in zip(texts, targets, gold_names, pred_names, confidence, metadata)
                       if g != p], columns=['source_id', 'text', 'target', 'gold', 'prediction', 'confidence'])
errors = errors.sort_values('confidence', ascending=False)
errors.to_csv(reports / 'apc_validation_errors.csv', index=False)
display(errors.head(15))
del apc
""")

md("""## 8. Final held-out evaluation

Report ATE exact and overlap F1, APC macro F1 on gold aspects, and end-to-end explicit tuple F1 separately. The implicit score uses a uniform document sample and a training-derived domain ontology, so include sample size, positive prevalence, and gold ontology coverage. The clothing test has no ATE/E2E score. Do not combine scores from annotation protocols into one unexplained number.
""")
code("""run_stage('eval')
results = json.loads((reports / 'results.json').read_text())
from IPython.display import Markdown
display(Markdown((reports / 'results.md').read_text()))
rows = []
for system, tests in results.items():
    for test_name, metrics in tests.items():
        rows.append({'system': system, 'test': test_name,
                     'ATE_exact_F1': metrics['ate']['exact']['f1'] if metrics['ate'] else None,
                     'APC_gold_macro_F1': metrics['apc']['macro_f1'],
                     'E2E_overlap_F1': metrics['e2e']['overlap_f1'] if metrics['e2e'] else None,
                     'implicit_subset_F1': metrics['implicit']['f1'] if metrics.get('implicit') else None})
display(pd.DataFrame(rows).round(3))
""")

md("""## 9. Export models and a reviewable report bundle

The demo selects examples from a public held-out split at a fixed seed; it is an illustration, not an additional benchmark. The report bundle contains environment details, revision identifiers, data quality reports, EDA tables, validation errors, training summaries, and final evaluation.
""")
code("""run_stage('export')
display(Markdown((reports / 'demo.md').read_text()))
from zipfile import ZipFile, ZIP_DEFLATED
bundle = WORK / 'turkish_absa_report_bundle.zip'
include = ['data/splits/manifest.json', 'reports/environment.txt',
           'reports/upstream_revisions.json', 'reports/data_audit.json',
           'reports/review_queue.jsonl',
           'reports/cleaning_report.json', 'reports/eda_report.json',
           'reports/eda_overview.png', 'reports/apc_validation_errors.csv',
           'reports/results.json', 'reports/results.md', 'reports/demo.md',
           'models/ate_full/train_summary.json', 'models/apc_full/train_summary.json']
include += [str(p.relative_to(WORK)) for p in reports.glob('eda_*.csv')]
with ZipFile(bundle, 'w', ZIP_DEFLATED) as archive:
    for rel in include:
        path = WORK / rel
        if path.exists(): archive.write(path, rel)
print('Model archive:', WORK / 'turkish_absa_models.zip')
print('Report bundle:', bundle)
""")

md("""## Reporting guidance

State the exact data counts, alignment and cleaning exclusions, held-out scores per dataset, GPU and package versions, seed, and known annotation limitations. The ATE threshold remains fixed and uncalibrated. Improvement claims require repeated seeds, paired uncertainty estimates, a controlled ablation, and independent review of difficult Turkish expressions. Do not present a quick run or validation-selected score as a final test result.
""")

for i, cell in enumerate(CELLS):
    cell["id"] = f"absa-{i:02d}"
nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
serialized = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
if re.search(r"[\u0600-\u06ff]", serialized):
    raise ValueError("Notebook contains Arabic-script characters")
OUT.write_text(serialized, encoding="utf-8")
print(f"Wrote {OUT} with {len(CELLS)} cells ({OUT.stat().st_size:,} bytes)")
