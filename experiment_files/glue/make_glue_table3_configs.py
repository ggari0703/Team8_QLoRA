from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path("reproduce_table3/configs/glue_table3")
MANIFEST_PATH = Path("reproduce_table3/configs/glue_table3_manifest.yaml")
RESULTS_CSV = "reproduce_table3/results/table3_glue_table3.csv"
MODEL_NAME = "FacebookAI/roberta-large"


# RoBERTa-large GLUE runs are configured as full-data sanity-to-scale runs.
# The same task-level training settings are applied to BF16, LoRA, and QLoRA.
GLUE_TASK_SETTINGS = {
    "cola": {"learning_rate": 2e-4, "num_train_epochs": 20, "max_length": 128},
    "sst2": {"learning_rate": 4e-4, "num_train_epochs": 10, "max_length": 128},
    "mrpc": {"learning_rate": 3e-4, "num_train_epochs": 20, "max_length": 128},
    "qqp": {"learning_rate": 3e-4, "num_train_epochs": 10, "max_length": 128},
    "stsb": {"learning_rate": 2e-4, "num_train_epochs": 20, "max_length": 128},
    "mnli": {"learning_rate": 3e-4, "num_train_epochs": 3, "max_length": 128},
    "qnli": {"learning_rate": 2e-4, "num_train_epochs": 3, "max_length": 128},
    "rte": {"learning_rate": 4e-4, "num_train_epochs": 20, "max_length": 128},
}

METHODS = {
    "bf16": {
        "lora": {"enabled": False, "r": "", "alpha": "", "dropout": ""},
        "quantization": {"load_in_4bit": False, "load_in_8bit": False, "quant_type": "none", "double_quant": False},
    },
    "lora_bf16": {
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["query", "value"]},
        "quantization": {"load_in_4bit": False, "load_in_8bit": False, "quant_type": "none", "double_quant": False},
    },
    "qlora_int8": {
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["query", "value"]},
        "quantization": {"load_in_4bit": False, "load_in_8bit": True, "quant_type": "int8", "double_quant": False},
    },
    "qlora_fp4": {
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["query", "value"]},
        "quantization": {"load_in_4bit": True, "load_in_8bit": False, "quant_type": "fp4", "double_quant": False},
    },
}


def config_for(task: str, method: str) -> dict:
    settings = GLUE_TASK_SETTINGS[task]
    method_settings = METHODS[method]
    return {
        "task_group": "roberta_glue",
        "dataset_name": f"glue/{task}",
        "glue_task": task,
        "model_name_or_path": MODEL_NAME,
        "method": method,
        "seed": 42,
        "max_length": settings["max_length"],
        "results_csv": RESULTS_CSV,
        "lora": method_settings["lora"],
        "quantization": method_settings["quantization"],
        "training": {
            "output_dir": f"reproduce_table3/outputs/glue_table3/{task}_{method}",
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": settings["learning_rate"],
            "num_train_epochs": settings["num_train_epochs"],
            "max_steps": -1,
            "logging_steps": 100,
            "save_strategy": "no",
            "optim": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.06,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "bf16": True,
        },
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_paths: list[str] = []
    for task in GLUE_TASK_SETTINGS:
        for method in METHODS:
            path = CONFIG_DIR / f"{task}_{method}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_for(task, method), f, sort_keys=False)
            manifest_paths.append(str(path))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"configs": manifest_paths}, f, sort_keys=False)

    print(f"Wrote {len(manifest_paths)} configs to {CONFIG_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
