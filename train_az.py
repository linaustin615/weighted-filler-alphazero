import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines import score_search_move, score_search_values
from game import FillerGame
from mcts_gumbel import run_gumbel_mcts
from policy import PolicyNet

SIZE, N_COLORS = 10, 6
N_SIMS = 150 #MCTS sims per move
BUFFER_SIZE = 10000 #replay buffer holds previous iterations of positions
GAMES_PER_ITER = 8
N_ITERS = 12 #expert-iteration rounds
EPOCHS = 4 #epochs per iteration
BS, LR = 128, 1e-3

TEACHER_GAMES = 100
TEACHER_DEPTH = 5
TEACHER_EPOCHS = 10

TRAIN_SEED = 0
TEACHER_SEED = 1
DEVELOPMENT_SEED = 10_001
DEVELOPMENT_GAMES = 100
EVAL_EVERY = 3

def self_play_game(game, net, rng, n_sims): #convert one self-play game into training data
    state = game.initial_state(rng) #fresh game
    records = []
    steps = 0
    max_steps = game.size * game.size * 4 #safety cap
    while not game.is_terminal(state) and steps < max_steps:
        move, policy = run_gumbel_mcts(game, net, state, n_sims, add_noise=True, rng=rng) #visit-count distribution
        pi = np.zeros(game.n_colors, dtype=np.float32) #MCTS-improved policy
        for action, p in policy.items():
            pi[action] = p #dict -> array
        records.append((game.encode(state), pi, state.player)) #record current player's perspective
        state = game.apply(state, move) #play Gumbel move and switch player
        steps += 1
    #label records with results from each player's view
    p1, p2 = game.score(state, 1), game.score(state, 2)
    winner = 1 if p1 > p2 else (2 if p2 > p1 else 0)
    data = []
    for enc, pi, player in records:
        z = (
            0.0 if winner == 0
            else 1.0 if winner == player
            else -1.0
        )
        data.append((enc, pi, z)) #add original training example

        permutation = rng.permutation(game.n_colors)
        augmented_enc, augmented_pi = permute_colors(enc, pi, permutation)

        augmented_enc = np.rot90(
            augmented_enc,
            2,
            axes=(1, 2)
        ).copy()
        data.append((augmented_enc, augmented_pi, z))

    return data

def teacher_game(game, rng, depth=5, policy_temperature=0.01):
    state = game.initial_state(rng)
    records = []

    while not game.is_terminal(state):
        action_values = score_search_values(game, state, depth)
        best_value = max(action_values.values())
        best_actions = [
            action for action, value in action_values.items()
            if np.isclose(value, best_value)]

        legal_actions = list(action_values)
        search_values = np.array([action_values[action] for action in legal_actions])

        scaled_values = (search_values - search_values.max()) / policy_temperature

        legal_probabilities = np.exp(scaled_values)
        legal_probabilities /= legal_probabilities.sum()

        target_policy = np.zeros(game.n_colors, dtype=np.float32)
        target_policy[legal_actions] = legal_probabilities

        action = int(rng.choice(best_actions))

        records.append((
            game.encode(state), #input
            target_policy, #policy head
            state.player #value head
        ))

        state = game.apply(state, action)

    p1, p2 = game.score(state, 1), game.score(state, 2)
    winner = 1 if p1 > p2 else 2 if p2 > p1 else 0

    data = []

    for encoded_state, target_policy, player in records:
        z = (
            0.0 if winner == 0
            else 1.0 if winner == player
            else -1.0
        )

        data.append((encoded_state, target_policy, z)) #add policy and value targets

        permutation = rng.permutation(game.n_colors)
        augmented_state, augmented_policy = permute_colors(encoded_state, target_policy, permutation)

        augmented_state = np.rot90(
            augmented_state,
            2,
            axes=(1, 2)
        ).copy()

        data.append((augmented_state, augmented_policy, z))

    return data


def train_on_data(net, opt, data, epochs, bs): #train on state, policy, and value targets
    encs = torch.tensor(np.array([d[0] for d in data])) #(M, C, H, W); M = batch size/number of positions
    pis = torch.tensor(np.array([d[1] for d in data])) #(M, n_colors)
    zs = torch.tensor([d[2] for d in data], dtype=torch.float32) #(M,)

    M = len(data)
    for _ in range(epochs):
        perm = torch.randperm(M)
        for i in range(0, M, bs): #batch
            idx = perm[i:i+bs]

            batch_encs = encs[idx]
            batch_pis = pis[idx]

            logits, value = net(batch_encs) #run one shuffled minibatch
            n_colors = logits.shape[1]

            colors = batch_encs[:, :n_colors]
            my_territory = batch_encs[:, n_colors:n_colors + 1]
            opp_territory = batch_encs[:, n_colors + 1:n_colors + 2]

            #check ownership + color masks
            illegal = (
                (colors * my_territory).sum(dim=(2, 3)) > 0
            ) | (
                (colors * opp_territory).sum(dim=(2, 3)) > 0
            )

            masked_logits = logits.masked_fill(illegal, -torch.inf)
            log_probs = torch.log_softmax(masked_logits, dim=1) #logits -> probs
            log_probs = log_probs.masked_fill(illegal, 0.0)

            policy_loss = -(batch_pis * log_probs).sum(dim=1).mean() #soft cross-entropy

            value_loss = F.mse_loss(value.squeeze(-1), zs[idx]) #(M, 1) -> (M,)
            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) #cap gradient size
            opt.step()


def greedy_move(game, state):
    return max(game.legal_moves(state),
               key=lambda action: game.immediate_gain(state, action)
    )

def raw_net_move(game, net, state):
    x = torch.tensor(game.encode(state)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits[0].numpy()
    legal = game.legal_moves(state)
    return legal[int(np.argmax(logits[legal]))]

def greedy_opp(game, state, rng):
    return greedy_move(game, state)

def random_opp(game, state, rng):
    legal = game.legal_moves(state)
    return legal[rng.integers(len(legal))]

def depth_three_opp(game, state, rng):
    return score_search_move(game, state, depth=3, rng=rng)

def eval_vs_greedy(game, net, rng, n_sims, games=60):
    if games % 2 != 0:
        raise ValueError("games must be even for paired evaluation")

    net.eval()
    wins = 0
    initial_states = [game.initial_state(rng) for _ in range(games // 2)]
    for initial_state in initial_states:
        for net_player in (1, 2):
            state = initial_state
            steps, max_steps = 0, game.size * game.size * 4

            while not game.is_terminal(state) and steps < max_steps:
                if state.player == net_player:
                    move, _ = run_gumbel_mcts(game, net, state, n_sims, add_noise=False)
                else:
                    move = greedy_move(game, state)

                state = game.apply(state, move)
                steps += 1

            if game.score(state, net_player) > game.score(state, 3 - net_player):
                wins += 1

    return wins / games

def eval_raw(game, net, rng, opponent, games=60):
    if games % 2 != 0: #play each board twice to remove bias
        raise ValueError("games must be even for paired evaluation")

    net.eval()
    wins = 0
    initial_states = [game.initial_state(rng) for _ in range(games // 2)]
    for initial_state in initial_states:
        for net_player in (1, 2):
            state = initial_state
            steps, max_steps = 0, game.size * game.size * 4

            while not game.is_terminal(state) and steps < max_steps:
                if state.player == net_player:
                    move = raw_net_move(game, net, state)
                else:
                    move = opponent(game, state, rng)

                state = game.apply(state, move)
                steps += 1

            if game.score(state, net_player) > game.score(state, 3 - net_player):
                wins += 1

    return wins / games

def permute_colors(encoded_state, policy, permutation): #permutation array
    augmented_state = encoded_state.copy()
    augmented_policy = np.zeros_like(policy)

    n_colors = len(policy)
    capture_offset = n_colors + 4

    for old_color in range(n_colors):
        new_color = permutation[old_color] #map each old color to its new index
        augmented_state[new_color] = encoded_state[old_color]
        augmented_state[capture_offset + new_color] = encoded_state[capture_offset + old_color] #move matching capture plane with color
        augmented_policy[new_color] = policy[old_color]

    return augmented_state, augmented_policy

def save_training_checkpoint(
        path,
        net,
        opt,
        next_iteration,
        buffer,
        train_rng,
        teacher_rng,
        best_raw_depth_three,
        config,
        metrics=None
):
    checkpoint = {
        "model_state": net.state_dict(),
        "optimizer_state": opt.state_dict(),
        "next_iteration": next_iteration,
        "buffer": list(buffer),
        "torch_rng_state": torch.get_rng_state(),
        "train_rng_state": train_rng.bit_generator.state,
        "teacher_rng_state": teacher_rng.bit_generator.state,
        "best_raw_depth_three": best_raw_depth_three,
        "config": config,
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_training_checkpoint(
        path,
        net,
        opt,
        buffer,
        train_rng,
        teacher_rng
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    net.load_state_dict(checkpoint["model_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    buffer.extend(checkpoint["buffer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    train_rng.bit_generator.state = checkpoint["train_rng_state"]
    if "teacher_rng_state" in checkpoint:
        teacher_rng.bit_generator.state = checkpoint["teacher_rng_state"]

    return (
        checkpoint["next_iteration"],
        checkpoint["best_raw_depth_three"],
        checkpoint.get("config"),
    )


def evaluate_development(game, net, seed, games):
    return {
        "raw_vs_random": eval_raw(
            game,
            net,
            np.random.default_rng(seed),
            random_opp,
            games=games,
        ),
        "raw_vs_greedy": eval_raw(
            game,
            net,
            np.random.default_rng(seed + 1),
            greedy_opp,
            games=games,
        ),
        "raw_vs_depth_three": eval_raw(
            game,
            net,
            np.random.default_rng(seed + 2),
            depth_three_opp,
            games=games,
        ),
    }


def append_metrics(path, metrics):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, sort_keys=True) + "\n")


def resolved_config(args):
    if args.preset == "smoke":
        defaults = {
            "teacher_games": 2,
            "teacher_epochs": 1,
            "iterations": 1,
            "games_per_iteration": 1,
            "simulations": 8,
            "epochs": 1,
            "development_games": 4,
            "eval_every": 1,
        }
    else:
        defaults = {
            "teacher_games": TEACHER_GAMES,
            "teacher_epochs": TEACHER_EPOCHS,
            "iterations": N_ITERS,
            "games_per_iteration": GAMES_PER_ITER,
            "simulations": N_SIMS,
            "epochs": EPOCHS,
            "development_games": DEVELOPMENT_GAMES,
            "eval_every": EVAL_EVERY,
        }

    config = {
        "size": args.size,
        "n_colors": args.n_colors,
        "hotspot_count": args.hotspot_count,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "teacher_depth": args.teacher_depth,
        "train_seed": args.seed,
        "teacher_seed": (
            args.teacher_seed
            if args.teacher_seed is not None
            else 1_000 + args.seed
        ),
        "development_seed": args.development_seed,
    }
    for key, default in defaults.items():
        value = getattr(args, key)
        config[key] = default if value is None else value
    return config


def validate_config(config):
    positive = (
        "size",
        "n_colors",
        "hotspot_count",
        "buffer_size",
        "batch_size",
        "learning_rate",
        "teacher_depth",
        "teacher_games",
        "teacher_epochs",
        "iterations",
        "games_per_iteration",
        "simulations",
        "epochs",
        "development_games",
        "eval_every",
    )
    for key in positive:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["development_games"] % 2:
        raise ValueError("development_games must be even")


def check_resume_config(config, saved_config):
    if saved_config is None:
        return
    fixed_keys = (
        "size",
        "n_colors",
        "hotspot_count",
        "buffer_size",
        "batch_size",
        "learning_rate",
        "teacher_depth",
        "train_seed",
        "teacher_seed",
    )
    mismatches = [
        key for key in fixed_keys
        if saved_config.get(key) != config.get(key)
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(f"resume configuration mismatch: {names}")


def run_training(config, output_dir, resume=None):
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.jsonl"
    if resume is None and (config_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains a run; resume it or use a new directory"
        )

    torch.manual_seed(config["train_seed"])
    game = FillerGame(
        size=config["size"],
        n_colors=config["n_colors"],
        hotspot_count=config["hotspot_count"],
    )
    net = PolicyNet(
        n_colors=config["n_colors"],
        size=config["size"],
    )
    opt = torch.optim.Adam(
        net.parameters(),
        lr=config["learning_rate"],
    )
    teacher_rng = np.random.default_rng(config["teacher_seed"])
    train_rng = np.random.default_rng(config["train_seed"])
    buffer = deque(maxlen=config["buffer_size"])
    best_raw_depth_three = -1.0
    start_iteration = 0

    if resume is not None:
        (
            start_iteration,
            best_raw_depth_three,
            saved_config,
        ) = load_training_checkpoint(
            resume,
            net,
            opt,
            buffer,
            train_rng,
            teacher_rng,
        )
        check_resume_config(config, saved_config)
        print(f"resumed at iteration {start_iteration}", flush=True)
    else:
        teacher_data = []
        for teacher_index in range(config["teacher_games"]):
            teacher_data += teacher_game(
                game,
                teacher_rng,
                depth=config["teacher_depth"],
            )
            if (teacher_index + 1) % 10 == 0:
                print(
                    f"teacher games {teacher_index + 1}/"
                    f"{config['teacher_games']}",
                    flush=True,
                )

        buffer.extend(teacher_data)
        net.train()
        train_on_data(
            net,
            opt,
            list(buffer),
            config["teacher_epochs"],
            config["batch_size"],
        )
        net.eval()
        teacher_metrics = evaluate_development(
            game,
            net,
            config["development_seed"],
            config["development_games"],
        )
        teacher_metrics.update({
            "stage": "teacher",
            "iteration": 0,
            "new_examples": len(teacher_data),
            "buffer_size": len(buffer),
        })
        best_raw_depth_three = teacher_metrics["raw_vs_depth_three"]
        append_metrics(metrics_path, teacher_metrics)
        save_training_checkpoint(
            output_dir / "teacher_checkpoint.pt",
            net,
            opt,
            0,
            buffer,
            train_rng,
            teacher_rng,
            best_raw_depth_three,
            config,
            teacher_metrics,
        )
        save_training_checkpoint(
            output_dir / "best_development_checkpoint.pt",
            net,
            opt,
            0,
            buffer,
            train_rng,
            teacher_rng,
            best_raw_depth_three,
            config,
            teacher_metrics,
        )
        print(
            f"teacher bootstrap examples {len(teacher_data)}  "
            f"raw_vs_greedy {teacher_metrics['raw_vs_greedy']:.1%}  "
            f"raw_vs_depth_three "
            f"{teacher_metrics['raw_vs_depth_three']:.1%}",
            flush=True,
        )

    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for iteration in range(start_iteration, config["iterations"]):
        net.eval()
        new_data = []
        for game_index in range(config["games_per_iteration"]):
            new_data += self_play_game(
                game,
                net,
                train_rng,
                config["simulations"],
            )
            print(
                f"iteration {iteration + 1}/{config['iterations']}  "
                f"self-play {game_index + 1}/"
                f"{config['games_per_iteration']}",
                flush=True,
            )

        buffer.extend(new_data)
        net.train()
        train_on_data(
            net,
            opt,
            list(buffer),
            config["epochs"],
            config["batch_size"],
        )

        should_evaluate = (
            (iteration + 1) % config["eval_every"] == 0
            or iteration + 1 == config["iterations"]
        )
        metrics = {
            "stage": "self_play",
            "iteration": iteration + 1,
            "new_examples": len(new_data),
            "buffer_size": len(buffer),
        }
        if should_evaluate:
            net.eval()
            metrics.update(evaluate_development(
                game,
                net,
                config["development_seed"],
                config["development_games"],
            ))
            raw_depth_three = metrics["raw_vs_depth_three"]
            if raw_depth_three > best_raw_depth_three:
                best_raw_depth_three = raw_depth_three
                save_training_checkpoint(
                    output_dir / "best_development_checkpoint.pt",
                    net,
                    opt,
                    iteration + 1,
                    buffer,
                    train_rng,
                    teacher_rng,
                    best_raw_depth_three,
                    config,
                    metrics,
                )
            print(
                f"iteration {iteration + 1}  data {len(new_data)}  "
                f"raw_vs_random {metrics['raw_vs_random']:.1%}  "
                f"raw_vs_greedy {metrics['raw_vs_greedy']:.1%}  "
                f"raw_vs_depth_three {raw_depth_three:.1%}  "
                f"best_depth_three {best_raw_depth_three:.1%}",
                flush=True,
            )
        else:
            print(
                f"iteration {iteration + 1}  data {len(new_data)}  "
                f"buffer {len(buffer)}",
                flush=True,
            )

        append_metrics(metrics_path, metrics)
        save_training_checkpoint(
            output_dir / "latest_checkpoint.pt",
            net,
            opt,
            iteration + 1,
            buffer,
            train_rng,
            teacher_rng,
            best_raw_depth_three,
            config,
            metrics,
        )

    torch.save(net.state_dict(), output_dir / "final_model.pt")
    best_checkpoint = torch.load(
        output_dir / "best_development_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(best_checkpoint["model_state"], output_dir / "best_model.pt")
    print(f"finished training in {output_dir}", flush=True)
    return net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the weighted 10x10 Filler policy",
    )
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--teacher-seed", type=int)
    parser.add_argument("--development-seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--n-colors", type=int, default=N_COLORS)
    parser.add_argument("--hotspot-count", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--batch-size", type=int, default=BS)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--teacher-depth", type=int, default=TEACHER_DEPTH)
    parser.add_argument("--teacher-games", type=int)
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--games-per-iteration", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--development-games", type=int)
    parser.add_argument("--eval-every", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    training_config = resolved_config(arguments)
    destination = (
        arguments.output_dir
        if arguments.output_dir is not None
        else Path("runs") / f"seed_{arguments.seed}"
    )
    run_training(
        training_config,
        destination,
        resume=arguments.resume,
    )
