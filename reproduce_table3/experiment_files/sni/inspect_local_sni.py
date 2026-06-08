from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_task_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def task_json_path(tasks_dir: Path, task_name: str) -> Path:
    filename = task_name if task_name.endswith(".json") else f"{task_name}.json"
    return tasks_dir / filename


def count_instances(repo_path: Path, task_names: list[str], eval_instances_per_task: int | None = None) -> tuple[int, int]:
    tasks_dir = repo_path / "tasks"
    total = 0
    missing = 0
    for task_name in task_names:
        path = task_json_path(tasks_dir, task_name)
        if not path.exists():
            missing += 1
            continue
        with open(path, "r", encoding="utf-8") as f:
            task_json = json.load(f)
        count = len(task_json.get("Instances", []))
        if eval_instances_per_task is not None and eval_instances_per_task > 0:
            count = min(count, eval_instances_per_task)
        total += count
    return total, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Count local AllenAI Natural Instructions split instances.")
    parser.add_argument("--repo", required=True, help="Path to allenai/natural-instructions checkout.")
    parser.add_argument("--split", default="default", help="Split directory under splits/.")
    parser.add_argument("--eval-instances-per-task", type=int, default=100)
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser()
    split_dir = repo_path / "splits" / args.split
    train_tasks = read_task_names(split_dir / "train_tasks.txt")
    test_tasks = read_task_names(split_dir / "test_tasks.txt")
    train_instances, missing_train = count_instances(repo_path, train_tasks)
    eval_instances, missing_eval = count_instances(repo_path, test_tasks, args.eval_instances_per_task)

    print(
        {
            "repo": str(repo_path),
            "split": args.split,
            "train_tasks": len(train_tasks),
            "test_tasks": len(test_tasks),
            "train_instances": train_instances,
            "eval_instances": eval_instances,
            "eval_instances_per_task": args.eval_instances_per_task,
            "missing_train_task_files": missing_train,
            "missing_eval_task_files": missing_eval,
        }
    )


if __name__ == "__main__":
    main()
