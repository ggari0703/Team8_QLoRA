from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


RESULT_COLUMNS = [
    "timestamp",
    "task_group",
    "dataset",
    "model_name",
    "method",
    "quant_type",
    "double_quant",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "max_train_samples",
    "max_eval_samples",
    "seed",
    "metric_name",
    "metric_value",
    "gpu_memory_peak_mb",
]


T5_TABLE3_MODELS = {
    "t5_80m": "google/t5-v1_1-small",
    "t5_250m": "google/t5-v1_1-base",
    "t5_780m": "google/t5-v1_1-large",
    "t5_3b": "google/t5-v1_1-xl",
    "t5_11b": "google/t5-v1_1-xxl",
}

ROBERTA_TABLE3_MODEL = "FacebookAI/roberta-large"


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_peak_memory_mb() -> str:
    if not torch.cuda.is_available():
        return ""
    return f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}"


def bf16_enabled(requested: bool) -> bool:
    if not requested:
        return False
    if not torch.cuda.is_available():
        print("[bf16 warning] CUDA is not available; Trainer bf16 mode is disabled.")
        return False
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        print("[bf16 warning] CUDA device does not report BF16 support; Trainer bf16 mode is disabled.")
        return False
    return True


def lora_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("lora", {}).get("enabled", True))


def infer_target_modules(model, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    names: set[str] = set()
    for module_name, module in model.named_modules():
        class_name = module.__class__.__name__
        is_linear = isinstance(module, torch.nn.Linear) or class_name in {"Linear4bit", "Linear8bitLt"}
        if not is_linear:
            continue
        leaf = module_name.split(".")[-1]
        if leaf not in exclude:
            names.add(leaf)
    return sorted(names)


def configured_target_modules(config: dict[str, Any], model, default: list[str], exclude: set[str] | None = None):
    target_modules = config.get("lora", {}).get("target_modules", default)
    if target_modules == "all-linear":
        inferred = infer_target_modules(model, exclude=exclude)
        if not inferred:
            raise ValueError("Could not infer any linear target modules for LoRA.")
        return inferred
    return target_modules


def append_result_row(
    csv_path: str | os.PathLike[str],
    config: dict[str, Any],
    metric_name: str,
    metric_value: float,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lora = config.get("lora", {})
    training = config.get("training", {})
    quant = config.get("quantization", {})

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_group": config.get("task_group", ""),
        "dataset": config.get("dataset_name") or config.get("glue_task", ""),
        "model_name": config.get("model_name_or_path", ""),
        "method": config.get("method", ""),
        "quant_type": quant.get("quant_type", "none"),
        "double_quant": quant.get("double_quant", False),
        "lora_r": lora.get("r", ""),
        "lora_alpha": lora.get("alpha", ""),
        "lora_dropout": lora.get("dropout", ""),
        "learning_rate": training.get("learning_rate", ""),
        "batch_size": training.get("per_device_train_batch_size", ""),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", ""),
        "max_train_samples": config.get("max_train_samples", ""),
        "max_eval_samples": config.get("max_eval_samples", ""),
        "seed": config.get("seed", ""),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "gpu_memory_peak_mb": cuda_peak_memory_mb(),
    }

    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_quantization_config(config: dict[str, Any]) -> tuple[Any | None, Any | None, str | None]:
    quant = config.get("quantization", {})
    load_in_4bit = bool(quant.get("load_in_4bit", False))
    load_in_8bit = bool(quant.get("load_in_8bit", False))

    if not load_in_4bit and not load_in_8bit:
        return None, torch.bfloat16, None

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Quantized QLoRA modes require CUDA for bitsandbytes in this framework. "
            "Use the LoRA BF16 config on CPU, or rerun QLoRA on a CUDA GPU."
        )

    from transformers import BitsAndBytesConfig

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", False)),
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if "device_map" in quant:
        device_map = quant["device_map"]
    else:
        local_rank = os.environ.get("LOCAL_RANK")
        device_map = {"": int(local_rank)} if local_rank is not None else {"": 0}

    return bnb_config, None, device_map


def select_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def repo_relative_path(path: str) -> str:
    return str(Path(path))
