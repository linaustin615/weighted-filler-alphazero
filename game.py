from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class WeightedFillerState:
    board: np.ndarray
    owner: np.ndarray
    cell_values: np.ndarray
    walls: np.ndarray
    player: int
    no_progress: int = 0

class FillerGame: #hold rules without mutable game state
    def __init__(self, size=10, n_colors=6, hotspot_count=4):
        self.size = size
        self.n_colors = n_colors
        self.hotspot_count = hotspot_count
        self.no_progress_limit = 6

    def initial_state(self, rng):
        board = rng.integers(0, self.n_colors, (self.size, self.size)).astype(np.int8)

        #sample distinct starting colors without bias
        first_color = int(board[-1, 0])
        other_colors = [
            color for color in range(self.n_colors)
            if color != first_color
        ]
        board[0, -1] = int(rng.choice(other_colors)) #set distinct opposing color

        owner = np.zeros((self.size, self.size), dtype=int)
        owner[-1, 0] = 1
        owner[0, -1] = 2
        cell_values = self._make_clustered_values(rng)
        walls = self._make_walls(rng)
        cell_values[walls] = 0
        owner[walls] = -1 #no player owns wall cells
        return WeightedFillerState(board, owner, cell_values, walls, player=1)

    def _anchor_color(self, board, player):
        return board[-1, 0] if player == 1 else board[0, -1]

    def legal_moves(self, state):
        c1, c2 = self._anchor_color(state.board, 1), self._anchor_color(state.board, 2)
        return [c for c in range(self.n_colors) if c != c1 and c != c2]

    @staticmethod
    def _neighbors(mask):
        n = np.zeros_like(mask)
        n[:-1] |= mask[1:]
        n[1:] |= mask[:-1]
        n[:, :-1] |= mask[:, 1:]
        n[:, 1:] |= mask[:, :-1]
        return n & ~mask

    def _make_clustered_values(self, rng):
            values = np.ones(
                (self.size, self.size),
                dtype=np.float32
            )

            for _ in range(self.hotspot_count):
                center_row = rng.integers(0, self.size)
                center_col = rng.integers(0, self.size)
                strength = rng.uniform(2.0, 8.0) #set additional hotspot value

                rows, cols = np.indices((self.size, self.size))
                distance_squared = (
                    (rows - center_row) ** 2
                    + (cols - center_col) ** 2
                )
                spread = rng.uniform(1.5, 3) #set hotspot width

                influence = strength * np.exp(
                    -distance_squared / (2 * spread ** 2)
                )
                values += influence

            values -= values.min() #set minimum to 0
            values /= values.max() #scale maximum to 1
            values = 1 + np.rint(values * 8) #map to integers from 1 through 9

            return values.astype(np.int8)

    def _make_walls(self, rng):
        walls = np.zeros(
            (self.size, self.size),
            dtype=bool
        )

        vertical = rng.random() < 0.5 #choose vertical or horizontal wall
        divider = rng.integers(3, self.size - 3) #place divider away from edges
        gate_positions = rng.choice( #sample two distinct gate cells
            np.arange(1, self.size - 1),
            size=2,
            replace=False
        )

        if vertical:
            walls[:, divider] = True
            walls[gate_positions, divider] = False
        else:
            walls[divider, :] = True
            walls[divider, gate_positions] = False

        #playable starting cells
        walls[-1, 0] = False
        walls[0, -1] = False

        return walls

    def encode(self, state):

            me = state.player
            opp = 3 - me
            planes = []
            for c in range(self.n_colors):
                planes.append(state.board == c)
            planes.append(state.owner == me)
            planes.append(state.owner == opp)
            planes.append(state.walls)
            planes.append(state.cell_values / 9.0) #normalize cell values

            for color in range(self.n_colors):
                planes.append(self._capture_mask(state, color))

            return np.stack(planes, axis=0).astype(np.float32) #return network-ready planes

    def _capture_mask(self, state, color):
        territory = state.owner == state.player
        while True:
            capture = (
                self._neighbors(territory)
                & (state.owner == 0)
                & (state.board == color)
            )

            if not capture.any():
                break

            territory |= capture

        return territory & (state.owner == 0)

    def apply(self, state, color):
        if color not in self.legal_moves(state):
            raise ValueError(f"illegal color: {color}")
        board, owner = state.board.copy(), state.owner.copy() #copy arrays to preserve input state
        capture = self._capture_mask(state, color)
        territory = (state.owner == state.player) | capture

        owner[territory] = state.player
        board[territory] = color
        captured = int(capture.sum())

        new_no_progress = 0 if captured > 0 else state.no_progress + 1

        return WeightedFillerState(
            board,
            owner,
            state.cell_values,
            state.walls,
            3 - state.player, #switch player
            new_no_progress
        )

    def is_terminal(self, state):
        if (state.owner != 0).all():
            return True
        if state.no_progress >= self.no_progress_limit:
            return True
        return False

    def _any_progress(self, state):
        legal = self.legal_moves(state)
        for player in (1, 2):
            frontier = self._neighbors(state.owner == player) & (state.owner == 0) #find adjacent neutral cells
            for color in legal:
                if (frontier & (state.board == color)).any():
                    return True
        return False

    def score(self, state, player):
        territory = state.owner == player
        return int(state.cell_values[territory].sum())

    def score_margin(self, state, player):
        return (
            self.score(state, player)
            - self.score(state, 3 - player)
        )

    def outcome(self, state):
        return int(
            np.sign(self.score_margin(state, state.player)) #collapse to -1, 0, or +1
        )

    def immediate_gain(self, state, color):
        score_before = self.score(state, state.player)
        next_state = self.apply(state, color)
        score_after = self.score(next_state, state.player)

        return score_after - score_before


if __name__ == "__main__":
    game = FillerGame()
    state = game.initial_state(np.random.default_rng(0))
    print(state.cell_values)
