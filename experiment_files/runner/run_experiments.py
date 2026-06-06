from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common.table3_common import load_config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.table3_common import load_config


def run_one(config_path: str) -> None:
    config = load_config(config_path)
    task_group = config.get("task_group")

    if task_group in {"t5_sni", "sni", "super_natural_instructions"}:
        try:
            from ..sni.train_t5_sni import run_from_config
        except ImportError:
            from sni.train_t5_sni import run_from_config
    elif task_group in {"roberta_glue", "glue"}:
        try:
            from ..glue.train_roberta_glue import run_from_config
        except ImportError:
            from glue.train_roberta_glue import run_from_config
    else:
        raise ValueError(f"Unknown task_group: {task_group!r}")

    run_from_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch Table 3 reproduction experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="YAML config path")
    group.add_argument("--manifest", help="YAML file containing a configs list")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.config:
        run_one(args.config)
        return

    manifest = load_config(args.manifest)
    failures: list[tuple[str, Exception]] = []
    for config_path in manifest.get("configs", []):
        try:
            print(f"[manifest] running {config_path}")
            run_one(config_path)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[manifest] failed {config_path}: {exc}", file=sys.stderr)
            failures.append((config_path, exc))

    if failures:
        print(f"[manifest] completed with {len(failures)} failure(s).", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
