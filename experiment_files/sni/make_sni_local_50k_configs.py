from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path("reproduce_table3/configs/sni_local_50k")
MANIFEST_PATH = Path("reproduce_table3/configs/sni_local_50k_manifest.yaml")
RESULTS_CSV = "reproduce_table3/results/table3_sni_local_50k.csv"
LOCAL_REPO = "reproduce_table3/data/natural-instructions"

MODELS = {
    "t5_80m": "google/t5-v1_1-small",
    "t5_250m": "google/t5-v1_1-base",
    "t5_780m": "google/t5-v1_1-large",
}

METHODS = {
    "lora_bf16": {
        "quantization": {"load_in_4bit": False, "load_in_8bit": False, "quant_type": "none", "double_quant": False},
        "lora": {"enabled": True, "r": 16, "alpha": 64, "dropout": 0.0, "target_modules": "all-linear"},
        "optim": "adamw_torch",
    },
    "qlora_nf4_dq": {
        "quantization": {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "nf4", "double_quant": True},
        "lora": {"enabled": True, "r": 16, "alpha": 64, "dropout": 0.0, "target_modules": "all-linear"},
        "optim": "paged_adamw_32bit",
    },
}


def config_for(size_key: str, method: str) -> dict:
    method_cfg = METHODS[method]
    return {
        "task_group": "t5_sni",
        "model_name_or_path": MODELS[size_key],
        "model_size_key": size_key,
        "dataset_name": "allenai/natural-instructions-local-50k",
        "local_sni_repo_path": LOCAL_REPO,
        "local_sni_split": "default",
        "local_sni_eval_instances_per_task": 100,
        "limit_local_sni_before_arrow": True,
        "method": method,
        "seed": 42,
        "max_train_samples": 50000,
        "max_eval_samples": 1000,
        "input_encoding": "tk_instruct_def_pos_2",
        "num_positive_examples": 2,
        "source_max_length": 1024,
        "target_max_length": 128,
        "generation_max_length": 64,
        "results_csv": RESULTS_CSV,
        "quantization": method_cfg["quantization"],
        "lora": method_cfg["lora"],
        "training": {
            "output_dir": f"reproduce_table3/outputs/sni_local_50k/{size_key}_{method}",
            "learning_rate": 1.0e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "max_steps": -1,
            "logging_steps": 100,
            "save_strategy": "no",
            "optim": method_cfg["optim"],
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "bf16": True,
        },
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for size_key in MODELS:
        for method in METHODS:
            path = CONFIG_DIR / f"{size_key}_{method}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_for(size_key, method), f, sort_keys=False)
            paths.append(str(path))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"configs": paths}, f, sort_keys=False)

    print(f"Wrote {len(paths)} configs to {CONFIG_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
