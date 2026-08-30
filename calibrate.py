import argparse
import json
from pathlib import Path

from evaluate import evaluate_opponent, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select gain scale using development boards only",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=(45.0, 50.0, 55.0, 60.0, 65.0),
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games % 2:
        raise ValueError("games must be even")

    _, _, checkpoint_config = load_model(args.checkpoint)
    development_seed = int(checkpoint_config.get("development_seed", 10_001))
    candidates = []
    for scale in args.scales:
        game, net, _ = load_model(args.checkpoint, gain_scale=scale)
        greedy = evaluate_opponent(
            game,
            net,
            "greedy",
            args.games,
            (development_seed + 1,),
        )
        depth_three = evaluate_opponent(
            game,
            net,
            "depth3",
            args.games,
            (development_seed + 2,),
        )
        objective = 0.5 * (
            greedy["match_score"] + depth_three["match_score"]
        )
        candidate = {
            "gain_scale": scale,
            "greedy_match_score": greedy["match_score"],
            "depth3_match_score": depth_three["match_score"],
            "mean_match_score": objective,
        }
        candidates.append(candidate)
        print(
            f"gain={scale:g}  greedy={greedy['match_score']:.1%}  "
            f"depth3={depth_three['match_score']:.1%}  "
            f"mean={objective:.1%}",
            flush=True,
        )

    selected = max(candidates, key=lambda candidate: candidate["mean_match_score"])
    report = {
        "checkpoint": str(args.checkpoint),
        "development_seed": development_seed,
        "games_per_opponent": args.games,
        "selected_gain_scale": selected["gain_scale"],
        "selection_objective": "mean greedy and depth3 match score",
        "candidates": candidates,
    }
    output = (
        args.output
        if args.output is not None
        else args.checkpoint.parent / "gain_calibration.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"selected gain scale {selected['gain_scale']:g}", flush=True)


if __name__ == "__main__":
    main()
