import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize official Weighted Filler seed evaluations",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/three_seed_summary.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reports = []
    for seed in args.seeds:
        evaluation_path = args.runs_dir / f"seed_{seed}" / "official_evaluation.json"
        calibration_path = args.runs_dir / f"seed_{seed}" / "gain_calibration.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        reports.append({
            "seed": seed,
            "gain_scale": calibration["selected_gain_scale"],
            "results": evaluation["results"],
        })

    aggregate = {}
    for opponent in ("random", "greedy", "depth3"):
        scores = np.array([
            report["results"][opponent]["match_score"]
            for report in reports
        ])
        aggregate[opponent] = {
            "mean_match_score": float(scores.mean()),
            "minimum_match_score": float(scores.min()),
            "maximum_match_score": float(scores.max()),
            "seed_match_scores": [float(score) for score in scores],
        }
        print(
            f"{opponent}: mean={scores.mean():.1%}  "
            f"range=[{scores.min():.1%}, {scores.max():.1%}]",
            flush=True,
        )

    summary = {
        "selection": "best development checkpoint and development-selected gain",
        "seeds": reports,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
