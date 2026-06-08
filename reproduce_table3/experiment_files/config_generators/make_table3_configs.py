from __future__ import annotations

from pathlib import Path

import yaml

try:
    from ..common.table3_common import ROBERTA_TABLE3_MODEL, T5_TABLE3_MODELS
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.table3_common import ROBERTA_TABLE3_MODEL, T5_TABLE3_MODELS


ROOT = Path("reproduce_table3")
OUT = ROOT / "configs" / "table3_full"


def base_training(output_dir: str, batch_size: int = 1) -> dict:
    return {
        "output_dir": output_dir,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1,
        "max_steps": 10000,
        "logging_steps": 10,
        "save_strategy": "steps",
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.03,
        "max_grad_norm": 0.3,
        "gradient_checkpointing": True,
        "bf16": True,
    }


def t5_config(model_key: str, model_name: str, method: str, quant: dict, lora_enabled: bool) -> dict:
    return {
        "task_group": "t5_sni",
        "model_name_or_path": model_name,
        "model_size_key": model_key,
        "dataset_name": "jayelm/natural-instructions",
        "dataset_config_name": None,
        "eval_filter_column": "eval",
        "method": method,
        "seed": 42,
        "max_train_samples": None,
        "max_eval_samples": None,
        "source_max_length": 1024,
        "target_max_length": 128,
        "generation_max_length": 128,
        "results_csv": "reproduce_table3/results/table3_runs.csv",
        "quantization": quant,
        "lora": {
            "enabled": lora_enabled,
            "r": 16,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "training": base_training(f"reproduce_table3/outputs/table3_full/{model_key}_{method}"),
    }


def roberta_config(method: str, quant: dict, lora_enabled: bool) -> dict:
    cfg = {
        "task_group": "roberta_glue",
        "model_name_or_path": ROBERTA_TABLE3_MODEL,
        "glue_task": "sst2",
        "dataset_name": "glue/sst2",
        "method": method,
        "seed": 42,
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_length": 256,
        "num_labels": 2,
        "results_csv": "reproduce_table3/results/table3_runs.csv",
        "quantization": quant,
        "lora": {
            "enabled": lora_enabled,
            "r": 16,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "training": base_training(f"reproduce_table3/outputs/table3_full/roberta_large_{method}", batch_size=8),
    }
    return cfg


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    no_quant = {"load_in_4bit": False, "load_in_8bit": False, "quant_type": "none", "double_quant": False}
    q_int8 = {"load_in_4bit": False, "load_in_8bit": True, "quant_type": "int8", "double_quant": False}
    q_fp4 = {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "fp4", "double_quant": False}
    q_nf4_dq = {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "nf4", "double_quant": True}

    manifest = {"configs": []}
    t5_methods = [
        ("bf16", no_quant, False),
        ("bf16_replication", no_quant, False),
        ("lora_bf16", no_quant, True),
        ("qlora_int8", q_int8, True),
        ("qlora_fp4", q_fp4, True),
        ("qlora_nf4_dq", q_nf4_dq, True),
    ]
    for model_key, model_name in T5_TABLE3_MODELS.items():
        for method, quant, lora_enabled in t5_methods:
            path = OUT / f"{model_key}_{method}.yaml"
            write_yaml(path, t5_config(model_key, model_name, method, quant, lora_enabled))
            manifest["configs"].append(str(path))

    roberta_methods = [
        ("bf16", no_quant, False),
        ("bf16_replication", no_quant, False),
        ("lora_bf16", no_quant, True),
        ("qlora_int8", q_int8, True),
        ("qlora_fp4", q_fp4, True),
    ]
    for method, quant, lora_enabled in roberta_methods:
        path = OUT / f"roberta_large_sst2_{method}.yaml"
        write_yaml(path, roberta_config(method, quant, lora_enabled))
        manifest["configs"].append(str(path))

    write_yaml(ROOT / "configs" / "table3_full_manifest.yaml", manifest)
    print(f"Wrote {len(manifest['configs'])} configs under {OUT}")


if __name__ == "__main__":
    main()
