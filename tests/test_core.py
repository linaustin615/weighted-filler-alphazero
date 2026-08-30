import unittest

import numpy as np
import torch

from game import FillerGame
from mcts_gumbel import run_gumbel_mcts
from policy import PolicyNet
from train_az import permute_colors


class WeightedFillerTests(unittest.TestCase):
    def setUp(self):
        self.game = FillerGame(size=10, n_colors=6)

    def test_initial_state_is_reproducible(self):
        left = self.game.initial_state(np.random.default_rng(123))
        right = self.game.initial_state(np.random.default_rng(123))
        np.testing.assert_array_equal(left.board, right.board)
        np.testing.assert_array_equal(left.owner, right.owner)
        np.testing.assert_array_equal(left.cell_values, right.cell_values)
        np.testing.assert_array_equal(left.walls, right.walls)

    def test_apply_is_immutable_and_switches_player(self):
        state = self.game.initial_state(np.random.default_rng(456))
        board_before = state.board.copy()
        owner_before = state.owner.copy()
        next_state = self.game.apply(state, self.game.legal_moves(state)[0])
        np.testing.assert_array_equal(state.board, board_before)
        np.testing.assert_array_equal(state.owner, owner_before)
        self.assertEqual(next_state.player, 2)

    def test_encoding_and_network_shapes(self):
        states = [
            self.game.initial_state(np.random.default_rng(seed))
            for seed in (1, 2)
        ]
        encoded = np.array([self.game.encode(state) for state in states])
        self.assertEqual(encoded.shape, (2, 16, 10, 10))
        self.assertEqual(encoded.dtype, np.float32)
        net = PolicyNet(n_colors=6, size=10)
        logits, values = net(torch.tensor(encoded))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(tuple(values.shape), (2, 1))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_color_permutation_round_trip(self):
        state = self.game.initial_state(np.random.default_rng(789))
        encoded = self.game.encode(state)
        policy = np.arange(6, dtype=np.float32)
        policy /= policy.sum()
        permutation = np.array([2, 5, 1, 4, 0, 3])
        transformed, transformed_policy = permute_colors(
            encoded,
            policy,
            permutation,
        )
        restored, restored_policy = permute_colors(
            transformed,
            transformed_policy,
            np.argsort(permutation),
        )
        np.testing.assert_array_equal(restored, encoded)
        np.testing.assert_array_equal(restored_policy, policy)

    def test_mcts_returns_a_legal_distribution(self):
        rng = np.random.default_rng(987)
        state = self.game.initial_state(rng)
        net = PolicyNet(n_colors=6, size=10)
        action, policy = run_gumbel_mcts(
            self.game,
            net,
            state,
            n_sims=8,
            add_noise=False,
        )
        legal = self.game.legal_moves(state)
        self.assertIn(action, legal)
        self.assertEqual(set(policy), set(legal))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=6)
        self.assertTrue(all(probability >= 0.0 for probability in policy.values()))


if __name__ == "__main__":
    unittest.main()
