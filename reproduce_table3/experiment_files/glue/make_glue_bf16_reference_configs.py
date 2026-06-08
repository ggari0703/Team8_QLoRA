from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path("reproduce_table3/configs/glue_bf16_reference")
MANIFEST_PATH = Path("reproduce_table3/configs/glue_bf16_reference_manifest.yaml")
RESULTS_CSV = "reproduce_table3/results/table3_glue_bf16_reference_lr_sweep.csv"
MODEL_NAME = "FacebookAI/roberta-large"


# Accuracy-based GLUE tasks only. CoLA and STS-B are excluded because their
# primary metrics are Matthews correlation and correlation scores.
ACCURACY_TASKS = ("sst2", "mrpc", "qqp", "mnli", "qnli", "rte")
LEARNING_RATES = (1e-5, 2e-5, 3e-5)


def config_for(task: str, learning_rate: float) -> dict:
    lr_key = f"{learning_rate:.0e}".replace("-", "m")
    return {
        "task_group": "roberta_glue",
        "dataset_name": f"glue/{task}",
        "glue_task": task,
        "model_name_or_path": MODEL_NAME,
        "method": "bf16_reference",
        "seed": 42,
        "max_length": 128,
        "results_csv": RESULTS_CSV,
        "lora": {"enabled": False, "r": "", "alpha": "", "dropout": ""},
        "quantization": {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "quant_type": "none",
            "double_quant": False,
        },
        "training": {
            "output_dir": f"reproduce_table3/outputs/glue_bf16_reference/{task}_lr_{lr_key}",
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "num_train_epochs": 10,
            "max_steps": -1,
            "logging_steps": 100,
            "save_strategy": "no",
            "optim": "adamw_torch",
            "weight_decay": 0.1,
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
    for task in ACCURACY_TASKS:
        for learning_rate in LEARNING_RATES:
            lr_key = f"{learning_rate:.0e}".replace("-", "m")
            path = CONFIG_DIR / f"{task}_bf16_lr_{lr_key}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_for(task, learning_rate), f, sort_keys=False)
            manifest_paths.append(str(path))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"configs": manifest_paths}, f, sort_keys=False)

    print(f"Wrote {len(manifest_paths)} configs to {CONFIG_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")
    print(f"Results CSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
