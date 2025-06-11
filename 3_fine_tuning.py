# 3_fine_tuning.py  (integrated Dodge‑style sweep)
# python 3_fine_tuning.py --regime XX_XX
# case sensitive!

from __future__ import annotations

# ── Environment ────────────────────────────────────────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random, argparse, warnings, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig, XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

import transformers, inspect, pathlib
print("Transformers version :", transformers.__version__)
print("Loaded from          :", pathlib.Path(transformers.__file__).resolve())
print("TrainingArguments…   :", inspect.signature(transformers.TrainingArguments).parameters.keys())

# initialize
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/"data"/"processed"
TOKENIZER_NAME = "xlm-roberta-base"
MODEL_NAME     = "xlm-roberta-base"
REGIMES = ["EN_EN", "CN_CN", "CN_EN", "EN_CN", "Joint"]

en_train_path = DATA_DIR / "en_train.csv"
cn_train_path = DATA_DIR / "cn_train.csv"
joint_train_path = DATA_DIR / "joint_train.csv"
en_test_path = DATA_DIR / "en_test.csv"
cn_test_path = DATA_DIR / "cn_test.csv"
joint_test_path = DATA_DIR / "joint_test.csv"

DATA_FILES = {
    "EN_EN": {"train": en_train_path, "valid": en_test_path},
    "CN_CN": {"train": cn_train_path, "valid": cn_test_path},
    "CN_EN": {"train": cn_train_path, "valid": en_test_path},
    "EN_CN": {"train": en_train_path, "valid": cn_test_path},
    "Joint": {"train": joint_train_path, "valid": joint_test_path},
}

# helpers
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def macro_f1(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"f1": f1_score(labels, preds, average="macro")}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8")
    df["text"] = df["text"].fillna("")
    df = df.rename(columns={"label": "labels"})
    return df


def df_to_ds(df: pd.DataFrame, tokenizer, data_seed: int):
    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        enc["labels"] = batch["labels"]
        return enc
    return Dataset.from_pandas(df).shuffle(seed=data_seed).map(tok, batched=True, remove_columns=list(df.columns))

def downsample_df(df: pd.DataFrame, target_len: int, seed: int = 123):
    # return a df with at most target_len rows, keeping chronology intact
    if len(df) <= target_len:
        return df
    df = df.sample(n=target_len, random_state=seed)
    return df.reset_index(drop=True)


def init_model(num_labels: int, use_lora: bool):
    cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
    if use_lora:
        lcfg = LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, lcfg)
    return model

# training of one seed x shuffle combo
def train_combo(args, wi_seed: int, do_seed: int) -> Dict[str, Any]:
    set_seeds(wi_seed)
    tk = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER_NAME)

    # load CSV
    en_tr = load_csv(DATA_FILES["EN_EN"]["train"])
    en_va = load_csv(DATA_FILES["EN_EN"]["valid"])
    cn_tr = load_csv(DATA_FILES["CN_CN"]["train"])
    cn_va = load_csv(DATA_FILES["CN_CN"]["valid"])

    TARGET_EN = len(en_tr)          # 10 458
    TARGET_CN = len(cn_tr)          # 21 928

    # build regime-specific frames
    if args.regime == "EN_EN":
        train_df, valid_df = en_tr, en_va

    elif args.regime == "CN_CN":
        train_df, valid_df = cn_tr, cn_va

    elif args.regime == "CN_EN":          # stage-1 handled via --lora sweep; here = stage-2
        cn_bal = downsample_df(cn_tr, TARGET_EN, seed=do_seed)
        train_df = pd.concat([en_tr, cn_bal], ignore_index=True)
        valid_df = en_va

    elif args.regime == "EN_CN":
        cn_bal = downsample_df(cn_tr, TARGET_EN, seed=do_seed)
        train_df = pd.concat([en_tr, cn_bal], ignore_index=True)
        valid_df = cn_va

    elif args.regime.lower() == "joint":
        cn_bal = downsample_df(cn_tr, TARGET_EN, seed=do_seed)
        train_df = pd.concat([en_tr, cn_bal], ignore_index=True)
        en_va_bal = downsample_df(en_va, min(len(en_va), len(cn_va)//2), seed=do_seed)
        cn_va_bal = downsample_df(cn_va, len(en_va_bal), seed=do_seed)
        valid_df = pd.concat([en_va_bal, cn_va_bal], ignore_index=True)

    else:
        raise ValueError(f"Unknown regime {args.regime}")


    train_ds = df_to_ds(train_df, tk, data_seed=do_seed)
    valid_ds = df_to_ds(valid_df, tk, data_seed=do_seed)

    out_dir = Path(args.output_dir)/f"{args.regime}_WI{wi_seed}_DO{do_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(num_labels=train_df["labels"].nunique(), use_lora=args.lora)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=1,
        seed=do_seed,
        data_seed=do_seed,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        disable_tqdm=args.quiet,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=valid_ds,
        tokenizer=tk,
        data_collator=DataCollatorWithPadding(tk),
        compute_metrics=macro_f1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train(); metrics = trainer.evaluate()

    if args.save_model:
        model.save_pretrained(out_dir/"best_model")

    return {"wi": wi_seed, "do": do_seed, "f1": metrics.get("eval_f1"), "loss": metrics.get("eval_loss")}

# sweep
def run_sweep(args):
    results: List[Dict[str, Any]] = []
    total = len(args.weight_seeds)*len(args.data_seeds)
    idx = 1
    for wi in args.weight_seeds:
        for do in args.data_seeds:
            print(f"[Run {idx}/{total}] WI={wi} DO={do}")
            results.append(train_combo(args, wi, do))
            idx += 1
    df = pd.DataFrame(results)

    # robustness over seeds
    mu = df["f1"].mean()
    sigma = df["f1"].std(ddof=1)  # unbiased
    print(f"{args.regime}: {mu:.3f} ± {sigma:.3f}")
    summary = pd.DataFrame(
        {"regime": [args.regime], "mu_macro_f1": [mu], "sigma_macro_f1": [sigma]}
    )
    summary_csv = Path(args.output_dir) / f"{args.regime}_sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Saved μ ± σ →", summary_csv)

    csv_name = args.results_csv or f"{args.regime}_sweep_results.csv"
    out_csv = Path(args.output_dir) / csv_name
    df.to_csv(out_csv, index=False)
    print("Saved summary: ", out_csv)
    best = df.sort_values("f1", ascending=False).iloc[0]
    print("Best combo:", best.to_dict())

# CLI
def build_parser():
    p = argparse.ArgumentParser(description="Fine‑tuning with sensitivity sweep (processed data paths)")
    p.add_argument("--regime", choices=REGIMES, default="EN_EN", help="Training regime (default: EN_EN)")
    p.add_argument("--single", action="store_true", help="Run only first WI/DO combo (debug)")
    p.add_argument("--weight_seeds", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--data_seeds", nargs="+", type=int, default=[1,2,3])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--output_dir", type=str, default="experiments")
    p.add_argument("--results_csv", type=str, default="sweep_results.csv")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.single:
        _ = train_combo(args, wi_seed=args.weight_seeds[0], do_seed=args.data_seeds[0])
    else:
        run_sweep(args)
