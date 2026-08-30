import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or resume the three Weighted Filler training seeds",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--calibration-games", type=int, default=200)
    parser.add_argument("--evaluation-games", type=int, default=1_000)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    for seed in args.seeds:
        output_dir = args.runs_dir / f"seed_{seed}"
        final_model = output_dir / "final_model.pt"
        latest_checkpoint = output_dir / "latest_checkpoint.pt"
        if final_model.exists():
            print(f"seed {seed} already complete: {final_model}", flush=True)
        else:
            command = [
                sys.executable,
                str(project_dir / "train_az.py"),
                "--preset",
                args.preset,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            if latest_checkpoint.exists():
                command.extend(("--resume", str(latest_checkpoint)))
            print(f"starting seed {seed}", flush=True)
            subprocess.run(command, cwd=project_dir, check=True)

        if args.skip_evaluation:
            continue

        best_checkpoint = output_dir / "best_development_checkpoint.pt"
        calibration_path = output_dir / "gain_calibration.json"
        evaluation_path = output_dir / "official_evaluation.json"
        if not calibration_path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(project_dir / "calibrate.py"),
                    str(best_checkpoint),
                    "--games",
                    str(args.calibration_games),
                ],
                cwd=project_dir,
                check=True,
            )
        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        gain_scale = calibration["selected_gain_scale"]
        if not evaluation_path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(project_dir / "evaluate.py"),
                    str(best_checkpoint),
                    "--games",
                    str(args.evaluation_games),
                    "--gain-scale",
                    str(gain_scale),
                    "--output",
                    str(evaluation_path),
                ],
                cwd=project_dir,
                check=True,
            )

    if not args.skip_evaluation:
        summary_path = (
            Path("results/three_seed_summary.json")
            if args.runs_dir == Path("runs")
            else args.runs_dir / "summary.json"
        )
        subprocess.run(
            [
                sys.executable,
                str(project_dir / "summarize_results.py"),
                "--runs-dir",
                str(args.runs_dir),
                "--seeds",
                *(str(seed) for seed in args.seeds),
                "--output",
                str(summary_path),
            ],
            cwd=project_dir,
            check=True,
        )


if __name__ == "__main__":
    main()
