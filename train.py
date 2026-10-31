from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fabir.config import load_config
from fabir.training import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FABIR with five-fold cross-validation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cancer", choices=("blca", "brca", "gbmlgg", "luad", "ucec"))
    group.add_argument("--config", type=Path)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5))
    parser.add_argument("--force-osrc-pretrain", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--semantic-risk-scale", type=float)
    parser.add_argument("--pathology-to-genomics-scale", type=float)
    parser.add_argument("--genomics-to-pathology-scale", type=float)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or Path(__file__).parent / "configs" / f"{args.cancer}.yaml"
    config = load_config(config_path)
    overrides = {
        "epochs": args.epochs,
        "seed": args.seed,
        "semantic_risk_scale": args.semantic_risk_scale,
        "pathology_to_genomics_residual_scale": args.pathology_to_genomics_scale,
        "genomics_to_pathology_residual_scale": args.genomics_to_pathology_scale,
        "results_dir": args.results_dir.resolve() if args.results_dir else None,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    run_cross_validation(
        config,
        folds=args.folds,
        force_osrc_pretrain=args.force_osrc_pretrain,
    )


if __name__ == "__main__":
    main()
