from __future__ import annotations

import argparse
from pathlib import Path

try:
    from analysis.margin_modeling import run_all
except ImportError:  # pragma: no cover
    from margin_modeling import run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA + baseline regression models for Futu/MooMoo margin requirements."
    )
    parser.add_argument(
        "--input",
        default="Regression_data.csv",
        help="Path to Regression_data.csv",
    )
    parser.add_argument(
        "--out",
        default="artifacts/margin_analysis",
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--target-scale",
        choices=["decimal", "pct_points"],
        default="decimal",
        help="Model targets as decimals (0.30) or percent points (30).",
    )
    parser.add_argument(
        "--winsor",
        type=float,
        default=0.01,
        help="Winsorization level for numeric features (e.g. 0.01 clips to [1%,99%]). Set 0 to disable.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Max depth for the rule-extraction decision tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    run_all(
        input_path=Path(args.input),
        out_dir=Path(args.out),
        target_scale=args.target_scale,
        winsor_alpha=args.winsor,
        tree_max_depth=args.tree_depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
