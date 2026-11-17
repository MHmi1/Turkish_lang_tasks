"""Train ATE (TokenClassification) or APC (SequenceClassification) with ModernBERT-TR.

Designed to run one task per GPU on Kaggle 2xT4 (e.g. ATE on GPU 0 and APC on GPU 1 in parallel):
  python -m absa.train --task ate --train X.jsonl --val V.jsonl --out DIR
  python -m absa.train --task apc --train X.jsonl --val V.jsonl --out DIR
"""
import argparse
import inspect
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from .config import APC_I2L, APC_L2I, ATE_I2L, ATE_L2I, BASE_MODEL, model_revision
from .data import encode_apc, encode_ate, read_jsonl
from .metrics import apc_trainer_metrics, ate_trainer_metrics


class ListDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class WTrainer(Trainer):
    cw = None
    label_smoothing = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        lg = out.logits.float()
        w = self.cw.to(lg.device) if self.cw is not None else None
        loss = F.cross_entropy(
            lg.view(-1, lg.size(-1)),
            labels.view(-1),
            weight=w,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        return (loss, out) if return_outputs else loss


def _clean_cfg(cfg):
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    for attr in ("bos_token_id", "eos_token_id"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v >= getattr(cfg, "vocab_size", 1 << 30):
            setattr(cfg, attr, None)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ate", "apc"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--negative_per_doc", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    a = ap.parse_args()
    set_seed(a.seed)

    revision = model_revision(a.model)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision)
    tr_docs = read_jsonl(a.train)
    va_docs = read_jsonl(a.val)

    if a.task == "ate":
        # train ATE on docs that have explicit spans (plus 15% implicit-only docs as negative examples)
        tr_exp = [d for d in tr_docs if any(isinstance(x.get("start"), int) for x in d["aspects"])]
        tr_imp = [d for d in tr_docs if d.get("span_supervision", True)
                  and not any(isinstance(x.get("start"), int) for x in d["aspects"])]
        random.Random(a.seed).shuffle(tr_imp)
        tr_use = tr_exp + tr_imp[: max(1, len(tr_exp) // 8)]
        random.Random(a.seed).shuffle(tr_use)
        tr = encode_ate(tok, tr_use, a.max_len)
        va = encode_ate(tok, [d for d in va_docs if d.get("span_supervision", True)], a.max_len)
        cfg = _clean_cfg(AutoConfig.from_pretrained(a.model, revision=revision, num_labels=len(ATE_L2I), id2label=ATE_I2L, label2id=ATE_L2I))
        try:
            model = AutoModelForTokenClassification.from_pretrained(a.model, revision=revision, config=cfg, attn_implementation="sdpa")
        except Exception:
            model = AutoModelForTokenClassification.from_pretrained(a.model, revision=revision, config=cfg)
        collator = DataCollatorForTokenClassification(tok, padding=True)
        metric_fn = ate_trainer_metrics
        best_metric = "f1"
        c = Counter(l for it in tr for l in it["labels"] if l != -100)
        o = max(c[0], 1)
        cw = torch.tensor([1.0, min(5.0, (o / max(c[1], 1)) ** 0.35), min(5.0, (o / max(c[2], 1)) ** 0.35)], dtype=torch.float32)
        smoothing = 0.0
    else:
        tr, _ = encode_apc(tok, tr_docs, a.max_len, train=True,
                           negative_per_doc=a.negative_per_doc, seed=a.seed)
        va, _ = encode_apc(tok, va_docs, a.max_len, add_negatives=True,
                           negative_per_doc=1, seed=a.seed+1)
        cfg = _clean_cfg(AutoConfig.from_pretrained(a.model, revision=revision, num_labels=len(APC_L2I), id2label=APC_I2L, label2id=APC_L2I))
        try:
            model = AutoModelForSequenceClassification.from_pretrained(a.model, revision=revision, config=cfg, attn_implementation="sdpa")
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(a.model, revision=revision, config=cfg)
        collator = DataCollatorWithPadding(tok, padding=True)
        metric_fn = apc_trainer_metrics
        best_metric = "macro_f1"
        c = Counter(it["labels"] for it in tr)
        tot = max(sum(c.values()), 1)
        cw = torch.tensor([min(4.0, (tot / (len(APC_L2I) * max(c[i], 1))) ** 0.45)
                           for i in range(len(APC_L2I))], dtype=torch.float32)
        smoothing = 0.05

    if a.max_train:
        random.Random(0).shuffle(tr)
        tr = tr[:a.max_train]
    print(f"[train-{a.task}] train={len(tr)} val={len(va)}", flush=True)

    steps_per_epoch = max(1, math.ceil(len(tr) / (a.bs * a.grad_accum)))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    warmup_steps = max(1, int(total_steps * 0.06))
    ev = max(1, steps_per_epoch // 2)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8

    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    raw_kw = dict(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs * 2,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        fp16=cuda and not bf16,
        bf16=bf16,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=ev,
        eval_steps=ev,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=a.seed,
        group_by_length=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    raw_kw[eval_key] = "steps"
    args = TrainingArguments(**{k: v for k, v in raw_kw.items() if k in sig})

    t_sig = inspect.signature(Trainer.__init__).parameters
    t_kw = dict(
        model=model,
        args=args,
        train_dataset=ListDS(tr),
        eval_dataset=ListDS(va),
        data_collator=collator,
        compute_metrics=metric_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)],
    )
    if "processing_class" in t_sig:
        t_kw["processing_class"] = tok
    elif "tokenizer" in t_sig:
        t_kw["tokenizer"] = tok

    trainer = WTrainer(**t_kw)
    trainer.cw = cw
    trainer.label_smoothing = smoothing
    trainer.train()

    final = os.path.join(a.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    m = trainer.evaluate()
    json.dump({"val": m, "args": vars(a), "n_train": len(tr)}, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print(f"[train-{a.task}] done", m, flush=True)


if __name__ == "__main__":
    main()
