import numpy as np

def random_move(game, state, rng):
    legal_moves = game.legal_moves(state)
    return int(rng.choice(legal_moves))

def greedy_move(game, state, rng=None):
    legal_moves = game.legal_moves(state)
    gains = []
    for color in legal_moves:
        gains.append(game.immediate_gain(state, color))
    best_position = int(np.argmax(gains))
    return legal_moves[best_position]

def _search_value(game, state, depth):
    if game.is_terminal(state):
        return game.outcome(state)

    #stop and estimate unfinished position
    if depth == 0:
        total_value = max(int(state.cell_values.sum()), 1)
        return game.score_margin(state, state.player) / total_value #estimate score margin from current player's perspective

    best_value = -float("inf")

    for action in game.legal_moves(state):
        child = game.apply(state, action)
        value = -_search_value(game, child, depth - 1) #convert child value to current-player perspective

        if value > best_value:
            best_value = value

    return best_value

def score_search_values(game, state, depth=3):
    action_values = {}

    for action in game.legal_moves(state):
        child = game.apply(state, action)
        action_values[action] = -_search_value(game, child, depth - 1)

    return action_values

def score_search_move(game, state, depth=3, rng=None):
    action_values = score_search_values(game, state, depth)
    return max(action_values, key=action_values.get)
