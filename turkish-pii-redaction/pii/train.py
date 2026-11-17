"""Fine-tune ModernBERT-TR for PII token classification (single-GPU, fp16, T4-optimized).

python -m pii.train --train X.jsonl --val V.jsonl --out DIR
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
from transformers import (AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from .config import BASE_MODEL, BIO, I2L, L2I
from .data import chunk_doc, encode_docs, read_jsonl
from .metrics import token_span_metrics


class ListDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class WTrainer(Trainer):
    cw = None
    _cw_dev = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        lg = out.logits.float()
        if self._cw_dev is None or self._cw_dev.device != lg.device:
            self._cw_dev = self.cw.to(lg.device, non_blocking=True)
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)), labels.view(-1), weight=self._cw_dev,
                               ignore_index=-100)
        return (loss, out) if return_outputs else loss


def class_weights(items, power=0.35, cap=6.0):
    c = Counter(l for it in items for l in it["labels"] if l != -100)
    w = np.ones(len(BIO), np.float32)
    o = max(c[0], 1)
    for i in range(1, len(BIO)):
        w[i] = min(cap, max(1.0, (o / max(c[i], 1)) ** power))
    return torch.tensor(w)


def load_items(tok, path, max_len, chunk=True):
    docs = read_jsonl(path)
    if chunk:
        docs = [c for d in docs for c in chunk_doc(d)]
    items, trunc = encode_docs(tok, docs, max_len)
    return items, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap train docs")
    a = ap.parse_args()
    set_seed(a.seed)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(a.model)
    tr, t1 = load_items(tok, a.train, a.max_len)
    va, t2 = load_items(tok, a.val, a.max_len)
    if a.max_train:
        random.Random(0).shuffle(tr)
        tr = tr[:a.max_train]
    print(f"[train] train={len(tr)} val={len(va)} truncated={t1}/{t2} bs={a.bs}", flush=True)

    cfg = AutoConfig.from_pretrained(a.model, num_labels=len(BIO), id2label=I2L, label2id=L2I)
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v >= getattr(cfg, "vocab_size", 1 << 30):
            setattr(cfg, attr, None)
    try:
        model = AutoModelForTokenClassification.from_pretrained(a.model, config=cfg, attn_implementation="sdpa")
    except Exception:
        model = AutoModelForTokenClassification.from_pretrained(a.model, config=cfg)

    steps_per_epoch = max(1, math.ceil(len(tr) / a.bs))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    warmup_steps = max(1, int(total_steps * 0.06))
    ev = max(100, steps_per_epoch // 2)
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    num_workers = 2 if cuda else 0
    raw_kw = dict(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        fp16=cuda and not bf16,
        bf16=bf16,
        optim="adamw_torch_fused" if cuda else "adamw_torch",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=ev,
        eval_steps=ev,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f2",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=a.seed,
        group_by_length=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=cuda,
        dataloader_persistent_workers=(num_workers > 0),
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
        data_collator=DataCollatorForTokenClassification(tok, padding=True, pad_to_multiple_of=8 if cuda else None),
        compute_metrics=token_span_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)],
    )
    if "processing_class" in t_sig:
        t_kw["processing_class"] = tok
    elif "tokenizer" in t_sig:
        t_kw["tokenizer"] = tok
    trainer = WTrainer(**t_kw)
    trainer.cw = class_weights(tr)
    trainer.train()
    final = os.path.join(a.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    m = trainer.evaluate()
    summary = {
        "val": m,
        "args": vars(a),
        "n_train": len(tr),
        "n_val": len(va),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(a.out, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("[train] done", m, flush=True)


if __name__ == "__main__":
    main()
