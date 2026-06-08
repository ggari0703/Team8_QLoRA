from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from .eval_t5_rougel import compute_rougel
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
    from eval_t5_rougel import compute_rougel
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


TOY_TRAIN = [
    {
        "definition": "Translate the English sentence into Korean.",
        "input": "Hello.",
        "output": "안녕하세요.",
    },
    {
        "definition": "Classify the sentiment as positive or negative.",
        "input": "The movie was thoughtful and moving.",
        "output": "positive",
    },
    {
        "definition": "Answer the question with a short phrase.",
        "input": "What color is the sky on a clear day?",
        "output": "blue",
    },
    {
        "definition": "Rewrite the sentence in a formal style.",
        "input": "Thanks for the help.",
        "output": "Thank you for your assistance.",
    },
]

TOY_EVAL = [
    {
        "definition": "Translate the English sentence into Korean.",
        "input": "Good morning.",
        "output": "좋은 아침입니다.",
    },
    {
        "definition": "Classify the sentiment as positive or negative.",
        "input": "The service was slow and frustrating.",
        "output": "negative",
    },
]


def _first_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_string(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("input", "output", "text", "label"):
            text = _first_string(value.get(key))
            if text:
                return text
    return str(value)


def _positive_examples(example: dict[str, Any], limit: int) -> list[tuple[str, str]]:
    positives: list[tuple[str, str]] = []
    for idx in range(limit):
        source = _first_string(
            example.get(f"pos_{idx}_input")
            or example.get(f"positive_{idx}_input")
            or example.get(f"Positive Example {idx + 1} Input")
        )
        target = _first_string(
            example.get(f"pos_{idx}_output")
            or example.get(f"positive_{idx}_output")
            or example.get(f"Positive Example {idx + 1} Output")
        )
        if source or target:
            positives.append((source, target))

    for key in ("positive_examples", "Positive Examples", "positives", "examples"):
        value = example.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, (list, tuple)):
            for item in value:
                if len(positives) >= limit:
                    break
                if isinstance(item, dict):
                    source = _first_string(item.get("input") or item.get("inputs") or item.get("source"))
                    target = _first_string(item.get("output") or item.get("outputs") or item.get("target"))
                    if source or target:
                        positives.append((source, target))
        if len(positives) >= limit:
            break

    return positives[:limit]


def format_instruction_example(example: dict[str, Any]) -> tuple[str, str]:
    definition = _first_string(
        example.get("definition")
        or example.get("Definition")
        or example.get("instruction")
        or example.get("instructions")
        or example.get("task")
    )
    source = _first_string(
        example.get("input")
        or example.get("inputs")
        or example.get("source")
        or example.get("Instance")
    )
    target = _first_string(
        example.get("output")
        or example.get("outputs")
        or example.get("target")
        or example.get("targets")
        or example.get("label")
    )

    prompt_parts = []
    if definition:
        prompt_parts.append(f"Definition: {definition.strip()}")

    input_encoding = str(example.get("_input_encoding", "definition_only"))
    num_positive_examples = int(example.get("_num_positive_examples", 2))
    if input_encoding in {"def_pos_2", "definition_positive_2", "tk_instruct_def_pos_2"}:
        for idx, (pos_input, pos_output) in enumerate(_positive_examples(example, num_positive_examples), start=1):
            prompt_parts.append(
                "\n".join(
                    [
                        f"Positive Example {idx} -",
                        f"Input: {pos_input.strip()}",
                        f"Output: {pos_output.strip()}",
                    ]
                )
            )

    if source:
        if input_encoding in {"def_pos_2", "definition_positive_2", "tk_instruct_def_pos_2"}:
            prompt_parts.append("Now complete the following example -")
        prompt_parts.append(f"Input: {source.strip()}")
    prompt_parts.append("Output:")
    return "\n".join(prompt_parts), target.strip()


def _read_task_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _task_json_path(tasks_dir: Path, task_name: str) -> Path:
    filename = task_name if task_name.endswith(".json") else f"{task_name}.json"
    return tasks_dir / filename


def _normalize_positive_examples(task_json: dict[str, Any]) -> list[dict[str, str]]:
    positives = task_json.get("Positive Examples", [])
    if not isinstance(positives, list):
        return []
    normalized = []
    for item in positives:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "input": _first_string(item.get("input")),
                "output": _first_string(item.get("output")),
            }
        )
    return normalized


def _iter_local_sni_examples(
    repo_path: Path,
    task_names: list[str],
    *,
    split: str,
    eval_instances_per_task: int,
    max_total_examples: int | None,
):
    tasks_dir = repo_path / "tasks"
    yielded = 0
    for task_name in task_names:
        task_path = _task_json_path(tasks_dir, task_name)
        if not task_path.exists():
            raise FileNotFoundError(f"Missing SNI task file: {task_path}")
        with open(task_path, "r", encoding="utf-8") as f:
            task_json = json.load(f)

        instances = task_json.get("Instances", [])
        if split == "validation" and eval_instances_per_task > 0:
            instances = instances[:eval_instances_per_task]

        definition = _first_string(task_json.get("Definition"))
        positives = _normalize_positive_examples(task_json)
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            target = _first_string(instance.get("output"))
            source = _first_string(instance.get("input"))
            if not target:
                continue
            yield {
                "task_name": task_name,
                "definition": definition,
                "input": source,
                "output": target,
                "positive_examples": positives,
            }
            yielded += 1
            if max_total_examples is not None and yielded >= max_total_examples:
                return


def load_local_sni_repo(config: dict[str, Any]) -> DatasetDict:
    repo_path = Path(config["local_sni_repo_path"]).expanduser()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"local_sni_repo_path does not exist: {repo_path}. "
            "Clone https://github.com/allenai/natural-instructions.git first."
        )

    split_name = config.get("local_sni_split", "default")
    split_dir = repo_path / "splits" / split_name
    train_tasks_file = Path(config.get("local_sni_train_tasks_file", split_dir / "train_tasks.txt"))
    eval_tasks_file = Path(config.get("local_sni_eval_tasks_file", split_dir / "test_tasks.txt"))
    eval_instances_per_task = int(config.get("local_sni_eval_instances_per_task", 100))

    train_tasks = _read_task_names(train_tasks_file)
    eval_tasks = _read_task_names(eval_tasks_file)
    max_train = config.get("max_train_samples") if bool(config.get("limit_local_sni_before_arrow", True)) else None
    max_eval = config.get("max_eval_samples") if bool(config.get("limit_local_sni_before_arrow", True)) else None

    train = Dataset.from_generator(
        lambda: _iter_local_sni_examples(
            repo_path,
            train_tasks,
            split="train",
            eval_instances_per_task=eval_instances_per_task,
            max_total_examples=max_train,
        )
    )
    validation = Dataset.from_generator(
        lambda: _iter_local_sni_examples(
            repo_path,
            eval_tasks,
            split="validation",
            eval_instances_per_task=eval_instances_per_task,
            max_total_examples=max_eval,
        )
    )
    return DatasetDict({"train": train, "validation": validation})


def load_sni_or_toy(config: dict[str, Any]) -> DatasetDict:
    dataset_name = config.get("dataset_name", "Muennighoff/natural-instructions")
    dataset_config_name = config.get("dataset_config_name")
    if bool(config.get("use_toy_dataset", False)) or dataset_name in {"toy", "toy_sni"}:
        return DatasetDict(
            {
                "train": Dataset.from_list(TOY_TRAIN),
                "validation": Dataset.from_list(TOY_EVAL),
            }
        )

    if config.get("local_sni_repo_path"):
        return load_local_sni_repo(config)

    try:
        if dataset_config_name:
            raw = load_dataset(dataset_name, dataset_config_name)
        else:
            raw = load_dataset(dataset_name)
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict({"train": raw})
        train = raw["train"] if "train" in raw else next(iter(raw.values()))
        eval_filter_column = config.get("eval_filter_column")
        if eval_filter_column and eval_filter_column in train.column_names:
            eval_ds = train.filter(lambda example: bool(example[eval_filter_column]))
            train = train.filter(lambda example: not bool(example[eval_filter_column]))
        else:
            eval_split_name = "validation" if "validation" in raw else "test" if "test" in raw else "train"
            eval_ds = raw[eval_split_name]
        return DatasetDict({"train": train, "validation": eval_ds})
    except Exception as exc:
        print(
            f"[dataset fallback] Failed to load {dataset_name!r}: {exc}. "
            "Using a tiny toy instruction dataset instead.",
            file=sys.stderr,
        )
        return DatasetDict(
            {
                "train": Dataset.from_list(TOY_TRAIN),
                "validation": Dataset.from_list(TOY_EVAL),
            }
        )


def preprocess_dataset(raw: DatasetDict, tokenizer, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    train_ds = select_subset(raw["train"], config.get("max_train_samples"))
    eval_ds = select_subset(raw["validation"], config.get("max_eval_samples"))
    source_max_length = int(config.get("source_max_length", 512))
    target_max_length = int(config.get("target_max_length", 128))

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        size = len(next(iter(batch.values())))
        examples = [{key: batch[key][idx] for key in batch} for idx in range(size)]
        for example in examples:
            example["_input_encoding"] = config.get("input_encoding", "definition_only")
            example["_num_positive_examples"] = int(config.get("num_positive_examples", 2))
        prompts, targets = zip(*(format_instruction_example(example) for example in examples))
        model_inputs = tokenizer(
            list(prompts),
            max_length=source_max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=list(targets),
            max_length=target_max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_train_cols = list(train_ds.column_names)
    remove_eval_cols = list(eval_ds.column_names)
    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=remove_train_cols)
    tokenized_eval = eval_ds.map(tokenize_batch, batched=True, remove_columns=remove_eval_cols)
    return tokenized_train, tokenized_eval


def build_model_and_tokenizer(config: dict[str, Any]):
    model_name = config.get("model_name_or_path", "t5-small")
    quantization_config, torch_dtype, device_map = build_quantization_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
    elif torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    if bool(config.get("training", {}).get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    if not lora_enabled(config):
        model.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"full finetune trainable params: {trainable} || all params: {total}")
        return model, tokenizer

    lora = config.get("lora", {})
    target_modules = configured_target_modules(
        config,
        model,
        default=["q", "v"],
        exclude={"lm_head"},
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
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
    raw = load_sni_or_toy(config)
    train_dataset, eval_dataset = preprocess_dataset(raw, tokenizer, config)

    training = config.get("training", {})
    output_dir = training.get("output_dir", "reproduce_table3/outputs/t5_sni")
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        max_steps=int(training.get("max_steps", -1)),
        logging_steps=int(training.get("logging_steps", 5)),
        save_strategy=training.get("save_strategy", "no"),
        report_to=[],
        optim=training.get("optim", "adamw_torch"),
        lr_scheduler_type=training.get("lr_scheduler_type", "constant"),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        predict_with_generate=True,
        generation_max_length=int(config.get("generation_max_length", 128)),
        bf16=bf16_enabled(bool(training.get("bf16", torch.cuda.is_available()))),
        fp16=False,
        seed=int(config.get("seed", 42)),
        remove_unused_columns=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    predictions = trainer.predict(eval_dataset)
    pred_ids = predictions.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    rouge_l = compute_rougel(decoded_preds, decoded_labels)

    append_result_row(
        config.get("results_csv", "reproduce_table3/results/table3_runs.csv"),
        config,
        "rougeL",
        rouge_l,
    )
    print({"rougeL": rouge_l})
    return rouge_l


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
