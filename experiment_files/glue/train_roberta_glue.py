from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from .eval_glue import compute_glue_metrics
    from ..common.table3_common import (
        append_result_row,
        bf16_enabled,
        build_quantization_config,
        configured_target_modules,
        lora_enabled,
        load_config,
        select_subset,
        set_seed,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from eval_glue import compute_glue_metrics
    from common.table3_common import (
        append_result_row,
        bf16_enabled,
        build_quantization_config,
        configured_target_modules,
        lora_enabled,
        load_config,
        select_subset,
        set_seed,
    )


GLUE_TEXT_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

TOY_GLUE_TRAIN = [
    {"sentence": "This film is excellent.", "label": 1},
    {"sentence": "The plot was dull.", "label": 0},
    {"sentence": "A sharp and funny story.", "label": 1},
    {"sentence": "I would not recommend it.", "label": 0},
]

TOY_GLUE_EVAL = [
    {"sentence": "A wonderful performance.", "label": 1},
    {"sentence": "The movie felt boring.", "label": 0},
]


class QuantizedBackboneSequenceClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.config = backbone.config
        self.config.num_labels = num_labels
        self.config.problem_type = "regression" if num_labels == 1 else "single_label_classification"
        hidden_size = self.config.hidden_size
        dropout_prob = getattr(self.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.backbone.gradient_checkpointing_enable()
            else:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=kwargs.get("output_hidden_states"),
            output_attentions=kwargs.get("output_attentions"),
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        pooled = torch.tanh(self.dense(pooled))
        pooled = self.dropout(pooled)
        logits = self.out_proj(pooled)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = torch.nn.MSELoss()(logits.view(-1), labels.float().view(-1))
            else:
                loss = torch.nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}")


def load_glue_or_toy(config: dict[str, Any]) -> DatasetDict:
    task = config.get("glue_task", "sst2")
    try:
        raw = load_dataset("glue", task)
        eval_split = "validation_matched" if task == "mnli" else "validation"
        return DatasetDict({"train": raw["train"], "validation": raw[eval_split]})
    except Exception as exc:
        print(
            f"[dataset fallback] Failed to load GLUE/{task}: {exc}. "
            "Using a tiny toy SST-2-like dataset instead.",
            file=sys.stderr,
        )
        return DatasetDict(
            {
                "train": Dataset.from_list(TOY_GLUE_TRAIN),
                "validation": Dataset.from_list(TOY_GLUE_EVAL),
            }
        )


def preprocess_dataset(raw: DatasetDict, tokenizer, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    task = config.get("glue_task", "sst2")
    text_key, text_pair_key = GLUE_TEXT_KEYS.get(task, ("sentence", None))
    max_length = int(config.get("max_length", 256))
    train_ds = select_subset(raw["train"], config.get("max_train_samples"))
    eval_ds = select_subset(raw["validation"], config.get("max_eval_samples"))

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        if text_pair_key:
            return tokenizer(
                batch[text_key],
                batch[text_pair_key],
                max_length=max_length,
                truncation=True,
            )
        return tokenizer(batch[text_key], max_length=max_length, truncation=True)

    remove_train_cols = [col for col in train_ds.column_names if col != "label"]
    remove_eval_cols = [col for col in eval_ds.column_names if col != "label"]
    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=remove_train_cols)
    tokenized_eval = eval_ds.map(tokenize_batch, batched=True, remove_columns=remove_eval_cols)
    return tokenized_train, tokenized_eval


def num_labels_for_config(config: dict[str, Any]) -> int:
    if "num_labels" in config:
        return int(config["num_labels"])
    task = config.get("glue_task", "sst2")
    return GLUE_NUM_LABELS.get(task, 2)


def build_model_and_tokenizer(config: dict[str, Any]):
    model_name = config.get("model_name_or_path", "FacebookAI/roberta-base")
    quantization_config, torch_dtype, device_map = build_quantization_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    num_labels = num_labels_for_config(config)
    model_kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
    elif torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if quantization_config is not None:
        backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        backbone = prepare_model_for_kbit_training(backbone)
        if bool(config.get("training", {}).get("gradient_checkpointing", False)):
            backbone.gradient_checkpointing_enable()
        if lora_enabled(config):
            lora = config.get("lora", {})
            target_modules = configured_target_modules(
                config,
                backbone,
                default=["query", "value"],
                exclude=set(),
            )
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(lora.get("r", 16)),
                lora_alpha=int(lora.get("alpha", 64)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=target_modules,
                bias="none",
            )
            backbone = get_peft_model(backbone, peft_config)
        model = QuantizedBackboneSequenceClassifier(backbone, num_labels=num_labels)
        model.is_parallelizable = True
        model.model_parallel = True
        print_trainable_parameters(model)
        return model, tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, **model_kwargs)
    if bool(config.get("training", {}).get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    if not lora_enabled(config):
        model.train()
        print_trainable_parameters(model)
        return model, tokenizer

    lora = config.get("lora", {})
    target_modules = configured_target_modules(
        config,
        model,
        default=["query", "value"],
        exclude={"classifier", "out_proj"},
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def run_from_config(config: dict[str, Any]) -> float:
    set_seed(int(config.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = build_model_and_tokenizer(config)
    raw = load_glue_or_toy(config)
    train_dataset, eval_dataset = preprocess_dataset(raw, tokenizer, config)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    task = config.get("glue_task", "sst2")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "stsb":
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=-1)
        return compute_glue_metrics(task, preds.tolist(), labels.tolist())

    training = config.get("training", {})
    args = TrainingArguments(
        output_dir=training.get("output_dir", "reproduce_table3/outputs/roberta_glue"),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        max_steps=int(training.get("max_steps", -1)),
        logging_steps=int(training.get("logging_steps", 5)),
        save_strategy=training.get("save_strategy", "no"),
        report_to=[],
        optim=training.get("optim", "adamw_torch"),
        weight_decay=float(training.get("weight_decay", 0.0)),
        lr_scheduler_type=training.get("lr_scheduler_type", "constant"),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        bf16=bf16_enabled(bool(training.get("bf16", torch.cuda.is_available()))),
        fp16=False,
        seed=int(config.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    ignored_metric_names = {
        "loss",
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "jit_compilation_time",
    }
    metric_values = {
        key.removeprefix("eval_"): float(value)
        for key, value in metrics.items()
        if key.startswith("eval_") and key.removeprefix("eval_") not in ignored_metric_names
    }
    for metric_name, metric_value in metric_values.items():
        append_result_row(
            config.get("results_csv", "reproduce_table3/results/table3_runs.csv"),
            config,
            metric_name,
            metric_value,
        )
    primary_metric = float(metric_values.get("glue_score", next(iter(metric_values.values()))))
    print(metric_values)
    return primary_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
