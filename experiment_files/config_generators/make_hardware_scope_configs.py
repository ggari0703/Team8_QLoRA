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
OUT = ROOT / "configs" / "hardware_scope"


def base_training(output_dir: str, batch_size: int, eval_batch_size: int | None = None, max_steps: int = 10000) -> dict:
    return {
        "output_dir": output_dir,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size or batch_size,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1,
        "max_steps": max_steps,
        "logging_steps": 25,
        "save_strategy": "no",
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.03,
        "max_grad_norm": 0.3,
        "gradient_checkpointing": True,
        "bf16": True,
    }


def t5_sni_training(output_dir: str, quant: dict) -> dict:
    is_quantized = bool(quant.get("load_in_4bit") or quant.get("load_in_8bit"))
    return {
        "output_dir": output_dir,
        "learning_rate": 0.00001,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 2,
        "max_steps": -1,
        "logging_steps": 25,
        "save_strategy": "no",
        "optim": "paged_adamw_32bit" if is_quantized else "adamw_torch",
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
    }


def roberta_training(output_dir: str, method: str, quant: dict) -> dict:
    is_full_bf16 = method.startswith("bf16")
    is_quantized = bool(quant.get("load_in_4bit") or quant.get("load_in_8bit"))
    return {
        "output_dir": output_dir,
        "learning_rate": 0.00002 if is_full_bf16 else 0.0001,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "logging_steps": 50,
        "save_strategy": "no",
        "optim": "paged_adamw_32bit" if is_quantized else "adamw_torch",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
    }


def t5_config(model_key: str, method: str, quant: dict, lora_enabled: bool) -> dict:
    batch_size = 1
    model_name = T5_TABLE3_MODELS[model_key]
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
        "input_encoding": "tk_instruct_def_pos_2",
        "num_positive_examples": 2,
        "source_max_length": 1024,
        "target_max_length": 128,
        "generation_max_length": 128,
        "results_csv": "reproduce_table3/results/table3_hardware_scope.csv",
        "quantization": quant,
        "lora": {
            "enabled": lora_enabled,
            "r": 16,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "training": t5_sni_training(
            f"reproduce_table3/outputs/hardware_scope/{model_key}_{method}",
            quant,
        ),
    }


def t5_pilot_config(model_key: str, method: str, quant: dict, lora_enabled: bool) -> dict:
    config = t5_config(model_key, method, quant, lora_enabled)
    config["max_train_samples"] = 16000
    config["max_eval_samples"] = 1000
    config["results_csv"] = "reproduce_table3/results/table3_hardware_scope_pilot.csv"
    config["training"]["output_dir"] = (
        f"reproduce_table3/outputs/hardware_scope_pilot/{model_key}_{method}"
    )
    config["training"]["num_train_epochs"] = 1
    config["training"]["max_steps"] = 1000
    return config


def roberta_config(method: str, quant: dict, lora_enabled: bool) -> dict:
    return {
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
        "results_csv": "reproduce_table3/results/table3_hardware_scope.csv",
        "quantization": quant,
        "lora": {
            "enabled": lora_enabled,
            "r": 16,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
        },
        "training": roberta_training(
            f"reproduce_table3/outputs/hardware_scope/roberta_large_sst2_{method}",
            method,
            quant,
        ),
    }


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    no_quant = {"load_in_4bit": False, "load_in_8bit": False, "quant_type": "none", "double_quant": False}
    q_int8 = {"load_in_4bit": False, "load_in_8bit": True, "quant_type": "int8", "double_quant": False}
    q_fp4 = {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "fp4", "double_quant": False}
    q_nf4_dq = {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "nf4", "double_quant": True}
    t5_methods = [
        ("bf16", no_quant, False),
        ("bf16_replication", no_quant, False),
        ("lora_bf16", no_quant, True),
        ("qlora_int8", q_int8, True),
        ("qlora_fp4", q_fp4, True),
        ("qlora_nf4_dq", q_nf4_dq, True),
    ]
    roberta_methods = [
        ("bf16", no_quant, False),
        ("bf16_replication", no_quant, False),
        ("lora_bf16", no_quant, True),
        ("qlora_int8", q_int8, True),
        ("qlora_fp4", q_fp4, True),
    ]

    manifest = {"configs": []}
    sni_manifest = {"configs": []}
    sni_pilot_manifest = {"configs": []}
    roberta_manifest = {"configs": []}
    for model_key in ["t5_80m", "t5_250m", "t5_780m"]:
        for method, quant, lora_enabled in t5_methods:
            path = OUT / f"{model_key}_{method}.yaml"
            write_yaml(path, t5_config(model_key, method, quant, lora_enabled))
            manifest["configs"].append(str(path))
            sni_manifest["configs"].append(str(path))

        for method, quant, lora_enabled in [
            ("lora_bf16", no_quant, True),
            ("qlora_nf4_dq", q_nf4_dq, True),
        ]:
            path = OUT / f"pilot_{model_key}_{method}.yaml"
            write_yaml(path, t5_pilot_config(model_key, method, quant, lora_enabled))
            sni_pilot_manifest["configs"].append(str(path))

    for method, quant, lora_enabled in roberta_methods:
        path = OUT / f"roberta_large_sst2_{method}.yaml"
        write_yaml(path, roberta_config(method, quant, lora_enabled))
        manifest["configs"].append(str(path))
        roberta_manifest["configs"].append(str(path))

    write_yaml(ROOT / "configs" / "hardware_scope_manifest.yaml", manifest)
    write_yaml(ROOT / "configs" / "hardware_scope_sni_manifest.yaml", sni_manifest)
    write_yaml(ROOT / "configs" / "hardware_scope_sni_pilot_manifest.yaml", sni_pilot_manifest)
    write_yaml(ROOT / "configs" / "hardware_scope_roberta_manifest.yaml", roberta_manifest)
    print(f"Wrote {len(manifest['configs'])} configs under {OUT}")


if __name__ == "__main__":
    main()
