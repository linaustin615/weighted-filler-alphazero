import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines import greedy_move, random_move, score_search_move
from game import FillerGame
from policy import PolicyNet


def load_model(path, gain_scale=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        config = payload.get("config") or {}
    else:
        state_dict = payload
        config = {}

    size = int(config.get("size", 10))
    n_colors = int(config.get("n_colors", 6))
    hotspot_count = int(config.get("hotspot_count", 4))
    net = PolicyNet(n_colors=n_colors, size=size)
    net.load_state_dict(state_dict)
    if gain_scale is not None:
        net.gain_scale.data.fill_(gain_scale)
    net.eval()
    game = FillerGame(
        size=size,
        n_colors=n_colors,
        hotspot_count=hotspot_count,
    )
    return game, net, config


def model_move(game, net, state):
    x = torch.tensor(game.encode(state)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(x)
    legal = game.legal_moves(state)
    return max(legal, key=lambda action: float(logits[0, action]))


def opponent_move(name, game, state, rng):
    if name == "random":
        return random_move(game, state, rng)
    if name == "greedy":
        return greedy_move(game, state)
    if name == "depth3":
        return score_search_move(game, state, depth=3)
    raise ValueError(f"unknown opponent: {name}")


def play_game(game, net, initial_state, net_player, opponent, rng):
    state = initial_state
    steps = 0
    max_steps = game.size * game.size * 4
    while not game.is_terminal(state) and steps < max_steps:
        if state.player == net_player:
            action = model_move(game, net, state)
        else:
            action = opponent_move(opponent, game, state, rng)
        state = game.apply(state, action)
        steps += 1

    net_score = game.score(state, net_player)
    opponent_score = game.score(state, 3 - net_player)
    if net_score > opponent_score:
        return 1
    if net_score < opponent_score:
        return -1
    return 0


def bootstrap_interval(pair_values, seed=91_337, samples=10_000):
    values = np.asarray(pair_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        resample = rng.integers(0, len(values), size=len(values))
        estimates[index] = values[resample].mean()
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def evaluate_opponent(game, net, opponent, games, board_seeds):
    divisor = 2 * len(board_seeds)
    if games % divisor:
        raise ValueError(
            f"games must be divisible by {divisor} for paired board seeds"
        )

    boards_per_seed = games // divisor
    wins = draws = losses = 0
    pair_scores = []
    pair_win_rates = []
    for seed_index, board_seed in enumerate(board_seeds):
        board_rng = np.random.default_rng(board_seed)
        for board_index in range(boards_per_seed):
            initial_state = game.initial_state(board_rng)
            pair_results = []
            for net_player in (1, 2):
                opponent_rng = np.random.default_rng(
                    board_seed
                    + 1_000_000
                    + board_index * 10
                    + net_player
                )
                result = play_game(
                    game,
                    net,
                    initial_state,
                    net_player,
                    opponent,
                    opponent_rng,
                )
                pair_results.append(result)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
            pair_scores.append(np.mean([
                1.0 if result > 0 else 0.5 if result == 0 else 0.0
                for result in pair_results
            ]))
            pair_win_rates.append(np.mean([
                1.0 if result > 0 else 0.0
                for result in pair_results
            ]))

    score_interval = bootstrap_interval(
        pair_scores,
        seed=91_337 + len(opponent),
    )
    win_interval = bootstrap_interval(
        pair_win_rates,
        seed=92_337 + len(opponent),
    )
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "strict_win_rate": wins / games,
        "strict_win_rate_ci95": list(win_interval),
        "match_score": (wins + 0.5 * draws) / games,
        "match_score_ci95": list(score_interval),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired held-out evaluation for Weighted Filler",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "greedy", "depth3"),
        default=("random", "greedy", "depth3"),
    )
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument(
        "--board-seeds",
        nargs="+",
        type=int,
        default=(20_001, 30_001, 40_001, 50_001, 60_001),
    )
    parser.add_argument("--gain-scale", type=float)
    parser.add_argument("--include-untrained", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    game, net, config = load_model(args.checkpoint, args.gain_scale)
    report = {
        "checkpoint": str(args.checkpoint),
        "gain_scale": float(net.gain_scale.detach()),
        "board_seeds": list(args.board_seeds),
        "config": config,
        "results": {},
    }
    for opponent in args.opponents:
        result = evaluate_opponent(
            game,
            net,
            opponent,
            args.games,
            args.board_seeds,
        )
        report["results"][opponent] = result
        print(
            f"{opponent}: {result['wins']}-{result['draws']}-"
            f"{result['losses']}  win={result['strict_win_rate']:.1%}  "
            f"score={result['match_score']:.1%}  "
            f"score_ci95=[{result['match_score_ci95'][0]:.1%}, "
            f"{result['match_score_ci95'][1]:.1%}]",
            flush=True,
        )

    if args.include_untrained:
        torch.manual_seed(0)
        untrained = PolicyNet(
            n_colors=game.n_colors,
            size=game.size,
        )
        untrained.eval()
        report["untrained_control"] = {}
        for opponent in args.opponents:
            result = evaluate_opponent(
                game,
                untrained,
                opponent,
                args.games,
                args.board_seeds,
            )
            report["untrained_control"][opponent] = result
            print(
                f"untrained_vs_{opponent}: {result['wins']}-"
                f"{result['draws']}-{result['losses']}  "
                f"win={result['strict_win_rate']:.1%}  "
                f"score={result['match_score']:.1%}  "
                f"score_ci95=[{result['match_score_ci95'][0]:.1%}, "
                f"{result['match_score_ci95'][1]:.1%}]",
                flush=True,
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
