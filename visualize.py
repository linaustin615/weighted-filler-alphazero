import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

from baselines import greedy_move
from game import FillerGame
from policy import PolicyNet


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "seed_1.pt"
DEFAULT_OUTPUT = ROOT / "trained_vs_untrained.gif"

BACKGROUND = "#07111f"
PANEL = "#101d31"
PANEL_EDGE = "#2b405f"
TEXT = "#edf4ff"
MUTED = "#91a4c0"
GRID = "#07111f"
TRAINED = "#45d6b0"
FRESH = "#ff6b7a"
GREEDY = "#ffb454"
HIGHLIGHT = "#ffe66d"
WALL = "#172235"
NEUTRAL = "#8a98ad"

COLOR_PALETTE = (
    "#ef476f",
    "#f78c3d",
    "#ffd166",
    "#3ec6a8",
    "#4d96e8",
    "#9b6de3",
)


def load_network(checkpoint, gain_scale):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model_state"] if "model_state" in payload else payload
    network = PolicyNet()
    network.load_state_dict(state_dict)
    network.gain_scale.data.fill_(gain_scale)
    network.eval()
    return network


def network_choice(game, network, state):
    encoded = torch.tensor(game.encode(state)).unsqueeze(0)
    with torch.no_grad():
        logits, value = network(encoded)
    legal = game.legal_moves(state)
    legal_logits = logits[0, legal]
    probabilities = torch.softmax(legal_logits, dim=0).numpy()
    preferences = np.zeros(game.n_colors, dtype=np.float32)
    preferences[legal] = probabilities / max(float(probabilities.max()), 1e-9)
    action = legal[int(torch.argmax(legal_logits))]
    return action, preferences, float(value.item())


def greedy_choice(game, state):
    legal = game.legal_moves(state)
    gains = np.array(
        [game.immediate_gain(state, action) for action in legal],
        dtype=np.float32,
    )
    preferences = np.zeros(game.n_colors, dtype=np.float32)
    if gains.max() > 0:
        preferences[legal] = gains / gains.max()
    else:
        preferences[legal] = 1.0
    return greedy_move(game, state), preferences


def record_episode(game, network, initial_state):
    state = initial_state
    states = [state]
    actions = []
    actors = []
    preferences = []
    values = []
    captures = []
    gains = []

    while not game.is_terminal(state):
        if state.player == 1:
            action, preference, value = network_choice(game, network, state)
            actor = "network"
        else:
            action, preference = greedy_choice(game, state)
            value = None
            actor = "greedy"

        capture = game._capture_mask(state, action)
        gain = int(state.cell_values[capture].sum())
        actions.append(action)
        actors.append(actor)
        preferences.append(preference)
        values.append(value)
        captures.append(capture)
        gains.append(gain)
        state = game.apply(state, action)
        states.append(state)

    return {
        "states": states,
        "actions": actions,
        "actors": actors,
        "preferences": preferences,
        "values": values,
        "captures": captures,
        "gains": gains,
        "network_scores": np.array([game.score(state, 1) for state in states]),
        "greedy_scores": np.array([game.score(state, 2) for state in states]),
    }


def make_board(axis, game, initial_state, title, title_color):
    axis.set_facecolor(PANEL)
    axis.set_xlim(-0.5, game.size - 0.5)
    axis.set_ylim(game.size - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticks(np.arange(-0.5, game.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, game.size, 1), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=1.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(1.4)

    board_image = axis.imshow(
        initial_state.board,
        cmap=ListedColormap(COLOR_PALETTE),
        vmin=-0.5,
        vmax=game.n_colors - 0.5,
        interpolation="nearest",
        zorder=1,
    )
    owner_image = axis.imshow(
        np.zeros((game.size, game.size, 4)),
        interpolation="nearest",
        zorder=2,
    )

    value_labels = []
    for row in range(game.size):
        for column in range(game.size):
            label = axis.text(
                column,
                row,
                "" if initial_state.walls[row, column] else str(initial_state.cell_values[row, column]),
                ha="center",
                va="center",
                color=TEXT,
                fontsize=8.5,
                fontweight="bold",
                zorder=5,
            )
            label.set_path_effects([
                path_effects.Stroke(linewidth=2.1, foreground="#07111f"),
                path_effects.Normal(),
            ])
            value_labels.append(label)

    capture_boxes = []
    for row in range(game.size):
        for column in range(game.size):
            box = Rectangle(
                (column - 0.43, row - 0.43),
                0.86,
                0.86,
                fill=False,
                edgecolor=HIGHLIGHT,
                linewidth=2.4,
                zorder=6,
                visible=False,
            )
            axis.add_patch(box)
            capture_boxes.append(box)

    title_text = axis.set_title(
        title,
        color=title_color,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground=BACKGROUND),
        path_effects.Normal(),
    ])
    score_text = axis.text(
        0.0,
        -0.068,
        "",
        transform=axis.transAxes,
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )
    action_text = axis.text(
        0.0,
        -0.116,
        "",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.4,
        ha="left",
        va="top",
    )

    return {
        "board": board_image,
        "owner": owner_image,
        "capture_boxes": capture_boxes,
        "score": score_text,
        "action": action_text,
    }


def owner_overlay(state):
    overlay = np.zeros((*state.owner.shape, 4), dtype=np.float32)
    trained_rgb = np.array(matplotlib.colors.to_rgb(TRAINED))
    greedy_rgb = np.array(matplotlib.colors.to_rgb(GREEDY))
    wall_rgb = np.array(matplotlib.colors.to_rgb(WALL))
    overlay[state.owner == 1, :3] = trained_rgb
    overlay[state.owner == 1, 3] = 0.48
    overlay[state.owner == 2, :3] = greedy_rgb
    overlay[state.owner == 2, 3] = 0.48
    overlay[state.walls, :3] = wall_rgb
    overlay[state.walls, 3] = 1.0
    return overlay


def update_board(artists, data, frame_index, game):
    state_index = min(frame_index, len(data["states"]) - 1)
    state = data["states"][state_index]
    artists["board"].set_data(state.board)
    artists["owner"].set_data(owner_overlay(state))

    capture = (
        data["captures"][state_index]
        if state_index < len(data["captures"])
        else np.zeros_like(state.walls)
    )
    for index, box in enumerate(artists["capture_boxes"]):
        row, column = divmod(index, game.size)
        box.set_visible(bool(capture[row, column]))

    network_score = game.score(state, 1)
    greedy_score = game.score(state, 2)
    artists["score"].set_text(
        f"network {network_score}   ·   greedy {greedy_score}"
    )

    if state_index < len(data["actions"]):
        action = data["actions"][state_index]
        actor = data["actors"][state_index]
        gain = data["gains"][state_index]
        actor_label = "raw network" if actor == "network" else "greedy opponent"
        artists["action"].set_color(TRAINED if actor == "network" else GREEDY)
        artists["action"].set_text(
            f"ply {state_index + 1}  ·  {actor_label} chooses {action}  ·  +{gain} value"
        )
    else:
        if network_score > greedy_score:
            outcome = "network wins"
            color = TRAINED
        elif network_score < greedy_score:
            outcome = "greedy wins"
            color = GREEDY
        else:
            outcome = "draw"
            color = MUTED
        artists["action"].set_color(color)
        artists["action"].set_text(f"final  ·  {outcome}  ·  no MCTS")


def style_choice_axis(axis, title, title_color):
    axis.set_facecolor(PANEL)
    axis.set_xlim(0, 1.08)
    axis.set_ylim(-0.65, 5.65)
    axis.set_yticks(range(6), labels=[f"color {index}" for index in range(6)])
    axis.invert_yaxis()
    axis.set_xticks((0, 0.5, 1.0))
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(axis="x", color=PANEL_EDGE, alpha=0.45, linewidth=0.8)
    axis.set_title(title, loc="left", color=title_color, fontsize=10.5, fontweight="bold")
    for spine in axis.spines.values():
        spine.set_color(PANEL_EDGE)

    bars = axis.barh(
        np.arange(6),
        np.zeros(6),
        color=COLOR_PALETTE,
        height=0.62,
        edgecolor="none",
        zorder=3,
    )
    label = axis.text(
        0.0,
        -0.24,
        "",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.5,
        ha="left",
        va="top",
    )
    return {"bars": bars, "label": label}


def update_choice(artists, data, frame_index):
    state_index = min(frame_index, len(data["states"]) - 1)
    if state_index >= len(data["actions"]):
        for bar in artists["bars"]:
            bar.set_width(0)
            bar.set_edgecolor("none")
        artists["label"].set_text("episode complete")
        return

    action = data["actions"][state_index]
    actor = data["actors"][state_index]
    preference = data["preferences"][state_index]
    for color, bar in enumerate(artists["bars"]):
        bar.set_width(float(preference[color]))
        if color == action:
            bar.set_edgecolor(HIGHLIGHT)
            bar.set_linewidth(2.2)
        else:
            bar.set_edgecolor("none")
            bar.set_linewidth(0)

    if actor == "network":
        value = data["values"][state_index]
        artists["label"].set_text(
            f"raw policy preference  ·  value estimate {value:+.2f}"
        )
    else:
        artists["label"].set_text("greedy immediate-gain preference")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a trained-versus-untrained Weighted Filler GIF",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gain-scale", type=float, default=50.0)
    parser.add_argument("--board-seed", type=int, default=40_001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    game = FillerGame()
    initial_state = game.initial_state(np.random.default_rng(args.board_seed))
    trained_network = load_network(args.checkpoint, args.gain_scale)
    torch.manual_seed(0)
    fresh_network = PolicyNet()
    fresh_network.eval()

    print("recording paired raw-policy episodes", flush=True)
    trained = record_episode(game, trained_network, initial_state)
    fresh = record_episode(game, fresh_network, initial_state)
    trained_final = trained["states"][-1]
    fresh_final = fresh["states"][-1]
    print(
        f"trained {game.score(trained_final, 1)}-{game.score(trained_final, 2)}  "
        f"untrained {game.score(fresh_final, 1)}-{game.score(fresh_final, 2)}",
        flush=True,
    )

    figure, axes = plt.subplot_mosaic(
        [
            ["trained", "trained", "fresh", "fresh", "trained_choice"],
            ["trained", "trained", "fresh", "fresh", "fresh_choice"],
            ["timeline", "timeline", "timeline", "timeline", "summary"],
        ],
        figsize=(17, 9),
        gridspec_kw={
            "width_ratios": (1.05, 1.05, 1.05, 1.05, 1.12),
            "height_ratios": (1.0, 1.0, 0.72),
        },
    )
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(
        left=0.035,
        right=0.98,
        top=0.84,
        bottom=0.075,
        wspace=0.34,
        hspace=0.72,
    )
    figure.suptitle(
        "Weighted Filler: what the policy learned",
        color=TEXT,
        fontsize=20,
        fontweight="bold",
        y=0.955,
    )
    figure.text(
        0.5,
        0.905,
        f"trained and fresh raw networks · identical held-out board · seed {args.board_seed} · same greedy opponent · no search",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )

    trained_artists = make_board(
        axes["trained"],
        game,
        initial_state,
        "trained policy · seed 1",
        TRAINED,
    )
    fresh_artists = make_board(
        axes["fresh"],
        game,
        initial_state,
        "fresh policy · seed 0",
        FRESH,
    )
    trained_choice = style_choice_axis(
        axes["trained_choice"],
        "trained decision",
        TRAINED,
    )
    fresh_choice = style_choice_axis(
        axes["fresh_choice"],
        "fresh decision",
        FRESH,
    )

    timeline = axes["timeline"]
    timeline.set_facecolor(PANEL)
    timeline.set_title("weighted territory over the episode", loc="left", color=TEXT, fontsize=11, fontweight="bold")
    timeline.set_xlabel("ply", color=MUTED, fontsize=9)
    timeline.set_ylabel("captured value", color=MUTED, fontsize=9)
    timeline.tick_params(colors=MUTED, labelsize=8)
    timeline.grid(color=PANEL_EDGE, alpha=0.42, linewidth=0.8)
    for spine in timeline.spines.values():
        spine.set_color(PANEL_EDGE)
    maximum_ply = max(len(trained["states"]), len(fresh["states"])) - 1
    maximum_score = max(
        trained["network_scores"].max(),
        trained["greedy_scores"].max(),
        fresh["network_scores"].max(),
        fresh["greedy_scores"].max(),
    )
    timeline.set_xlim(0, maximum_ply)
    timeline.set_ylim(0, maximum_score * 1.12)
    trained_network_line, = timeline.plot([], [], color=TRAINED, linewidth=2.7, label="trained network")
    trained_greedy_line, = timeline.plot([], [], color=GREEDY, linewidth=2.2, label="greedy in trained game")
    fresh_network_line, = timeline.plot([], [], color=FRESH, linewidth=2.4, linestyle="--", label="fresh network")
    fresh_greedy_line, = timeline.plot([], [], color=GREEDY, linewidth=2.0, linestyle="--", alpha=0.72, label="greedy in fresh game")
    timeline.legend(
        loc="upper left",
        frameon=False,
        labelcolor=TEXT,
        fontsize=8,
        ncol=2,
    )

    summary = axes["summary"]
    summary.set_facecolor(PANEL)
    summary.set_xticks([])
    summary.set_yticks([])
    for spine in summary.spines.values():
        spine.set_color(PANEL_EDGE)
    summary.text(
        0.06,
        0.87,
        "HELD-OUT EVIDENCE",
        transform=summary.transAxes,
        color=MUTED,
        fontsize=8.5,
        fontweight="bold",
    )
    summary.text(
        0.06,
        0.64,
        "trained · 3-seed mean",
        transform=summary.transAxes,
        color=TRAINED,
        fontsize=10.5,
        fontweight="bold",
    )
    summary.text(
        0.06,
        0.45,
        "56.7% vs greedy  ·  51.4% vs depth 3",
        transform=summary.transAxes,
        color=TEXT,
        fontsize=9.2,
    )
    summary.text(
        0.06,
        0.25,
        "fresh control · seed 0",
        transform=summary.transAxes,
        color=FRESH,
        fontsize=10.5,
        fontweight="bold",
    )
    summary.text(
        0.06,
        0.08,
        "50.1% vs greedy  ·  44.7% vs depth 3",
        transform=summary.transAxes,
        color=TEXT,
        fontsize=9.2,
    )

    legend_handles = [
        Patch(facecolor=TRAINED, label="network territory"),
        Patch(facecolor=GREEDY, label="greedy territory"),
        Patch(facecolor=NEUTRAL, label="neutral cell"),
        Patch(facecolor=WALL, label="wall"),
        Patch(facecolor="none", edgecolor=HIGHLIGHT, linewidth=2, label="next capture"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.405, 0.012),
        ncol=5,
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
    )
    figure.text(
        0.98,
        0.018,
        "single illustrative game · aggregate results are the scientific comparison",
        ha="right",
        color=MUTED,
        fontsize=8.2,
    )

    frame_count = max(len(trained["states"]), len(fresh["states"]))
    frames = list(range(frame_count))
    frames.extend([frames[-1]] * (args.fps * 2))

    def update(frame_index):
        update_board(trained_artists, trained, frame_index, game)
        update_board(fresh_artists, fresh, frame_index, game)
        update_choice(trained_choice, trained, frame_index)
        update_choice(fresh_choice, fresh, frame_index)

        trained_end = min(frame_index + 1, len(trained["states"]))
        fresh_end = min(frame_index + 1, len(fresh["states"]))
        trained_x = np.arange(trained_end)
        fresh_x = np.arange(fresh_end)
        trained_network_line.set_data(trained_x, trained["network_scores"][:trained_end])
        trained_greedy_line.set_data(trained_x, trained["greedy_scores"][:trained_end])
        fresh_network_line.set_data(fresh_x, fresh["network_scores"][:fresh_end])
        fresh_greedy_line.set_data(fresh_x, fresh["greedy_scores"][:fresh_end])

    animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=1000 / args.fps,
        blit=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(frames)} frames", flush=True)
    animation.save(
        args.output,
        writer=PillowWriter(fps=args.fps),
        dpi=args.dpi,
    )
    plt.close(figure)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
