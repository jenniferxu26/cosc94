import os
# allow duplicate OpenMP runtimes on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch

def train_sentiment_models(
    df_en: pd.DataFrame,
    df_cn: pd.DataFrame,
    output_dir: str = "./xlm_sentiment_models",
    cache_dir: str = "./model_cache",
) -> pd.DataFrame:

    # remap labels from {-1,0,1} → {0,1,2}
    label_map = {-1: 0, 0: 1, 1: 2}
    df_en = df_en.copy()
    df_cn = df_cn.copy()
    df_en["label"] = df_en["label"].map(label_map)
    df_cn["label"] = df_cn["label"].map(label_map)

    # pandas -> HF Dataset
    en_full = Dataset.from_pandas(df_en, preserve_index=False)
    cn_full = Dataset.from_pandas(df_cn, preserve_index=False)

    # tokenizer & collator
    model_checkpoint = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_checkpoint, cache_dir=cache_dir, local_files_only=False
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["query", "value"],
    )

    # preprocess function
    def preprocess(batch):
        texts = [str(t) for t in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=256)

    en_full = en_full.map(preprocess, batched=True)
    cn_full = cn_full.map(preprocess, batched=True)

    # keep only model inputs + label
    keep = ["input_ids", "attention_mask", "label", "timestamp", "text"]
    en_full = en_full.select_columns(keep)
    cn_full = cn_full.select_columns(keep)

    # split (80/20)
    # keep only model inputs + label
    keep = ["input_ids", "attention_mask", "label", "timestamp", "text"]
    en_full = en_full.select_columns(keep)
    cn_full = cn_full.select_columns(keep)

    # chronological 80 / 10 / 10 split
    en_split = en_full.train_test_split(test_size=0.2, seed=42)
    en_train, en_test = en_split["train"], en_split["test"]

    cn_split = cn_full.train_test_split(test_size=0.2, seed=42)
    cn_train_full, cn_test_full = cn_split["train"], cn_split["test"]

    # down-sampling helper
    def downsample(ds, target_len, seed=123):
        if len(ds) <= target_len:
            return ds
        # shuffle, then take the first `target_len` rows
        return ds.shuffle(seed=seed).select(range(target_len))

    TARGET_EN = len(en_train)  # 10 458 rows
    TARGET_CN = len(cn_train_full)  # 21 928 rows

    ## Regime-specific slices
    # EN → EN
    en_train_ENEN = en_train
    en_test_ENEN = en_test

    # CN → CN
    cn_train_CNCN = cn_train_full
    cn_test_CNCN = cn_test_full

    # CN → EN (all CN; balanced EN + CN)
    cn_stage1 = cn_train_full
    cn_balanced = downsample(cn_train_full, TARGET_EN)
    ft_CNEN = concatenate_datasets([en_train, cn_balanced]).shuffle(seed=42)

    # EN → CN
    en_stage1 = en_train
    ft_ENCN = concatenate_datasets([en_train, cn_balanced]).shuffle(seed=42)

    # joint EN+CN  (balanced mix)
    joint_train = concatenate_datasets([en_train, cn_balanced]).shuffle(seed=42)
    joint_test = concatenate_datasets(
        [en_test, downsample(cn_test_full, len(en_test))]
    ).shuffle(seed=42)

    # define regimes (pre_ds , ft_ds , ev_ds)
    regimes = [
        ("EN_EN", None, en_train_ENEN, en_test_ENEN),
        ("CN_CN", None, cn_train_CNCN, cn_test_CNCN),
        ("CN_EN", cn_stage1, ft_CNEN,   en_test_ENEN),  # eval on EN test
        ("EN_CN", en_stage1, ft_ENCN,   cn_test_CNCN),  # eval on CN test
        ("Joint", None,       joint_train, joint_test),
    ]


    # macro-F1 metric
    def compute_metrics(p):
        preds = p.predictions.argmax(axis=-1)
        return {"eval_macro_f1": f1_score(p.label_ids, preds, average="macro")}

    ## Training loop
    os.makedirs(output_dir, exist_ok=True)
    results = []
    num_labels = len(label_map)

    for name, pre_ds, ft_ds, ev_ds in regimes:
        print(f"\n=== Regime: {name} ===")

        # load base model
        base_model = XLMRobertaForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            local_files_only=False,
        )
        # wrap in LoRA
        model = get_peft_model(base_model, lora_config)

        # pre-training (optional)
        if pre_ds is not None:
            pre_args = TrainingArguments(
                output_dir=f"{output_dir}/{name}_pretrain",
                do_train=True,
                do_eval=False,
                per_device_train_batch_size=16,
                num_train_epochs=1,
                fp16=True,
                logging_steps=50,
                save_steps=200,
                save_total_limit=1,
                logging_dir=f"{output_dir}/{name}_pretrain/logs",
            )
            pre_trainer = Trainer(
                model=model,
                args=pre_args,
                train_dataset=pre_ds,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            pre_trainer.train()
            pre_trainer.save_model(f"{output_dir}/{name}_pretrain")  # save adapter

            # reload adapter
            base_model = XLMRobertaForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=num_labels,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )
            model = get_peft_model(base_model, lora_config)
            model.load_adapter(
                model_id=f"{output_dir}/{name}_pretrain",
                adapter_name="default"
            )

        # fine-tuning (3 epochs)
        ft_args = TrainingArguments(
            output_dir=f"{output_dir}/{name}",
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            fp16=True,
            gradient_checkpointing=False,
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            logging_dir=f"{output_dir}/{name}/logs",
        )
        trainer = Trainer(
            model=model,
            args=ft_args,
            train_dataset=ft_ds,
            eval_dataset=ev_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        f1 = metrics["eval_macro_f1"]
        print(f"[{name}] macro-F1 = {f1:.4f}")

        # per-class report
        pred_out = trainer.predict(ev_ds)
        y_true, y_pred = pred_out.label_ids, pred_out.predictions.argmax(axis=-1)

        from sklearn.metrics import classification_report
        import pandas as pd
        from pathlib import Path

        txt_report = classification_report(y_true, y_pred, digits=3)
        print("\nDetailed per-class metrics\n" + txt_report)

        df_report = pd.DataFrame(
            classification_report(y_true, y_pred, digits=3, output_dict=True)
        ).transpose()

        rep_csv = Path(output_dir) / f"{name}_class_report.csv"
        df_report.to_csv(rep_csv, index=True)
        print("Saved classification report: ", rep_csv)

        # save final model + LoRA adapter
        model.save_pretrained(f"{output_dir}/{name}")

        # save the fine-tuned classifier head
        head_state = {
            k: v.cpu()
            for k, v in model.base_model.classifier.state_dict().items()
        }
        torch.save(head_state, Path(output_dir) / name / "head_state.pt")

        results.append({"regime": name, "macro_f1": f1})

    # return results DataFrame
    return pd.DataFrame(results)


if __name__ == "__main__":
    en_csv = "data/processed/df_en_processed.csv"
    cn_csv = "data/processed/df_cn_processed.csv"

    print("Loading data ===")
    df_en = pd.read_csv(en_csv, encoding="utf-8")
    df_cn = pd.read_csv(cn_csv, encoding="utf-8")

    print("Starting training ===")
    results_df = train_sentiment_models(df_en, df_cn)

    # save results
    results_df.to_csv(f"sentiment_results.csv", index=False)
    print("\nResults:\n", results_df)