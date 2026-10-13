"""Agents that pick a move for a 2048 board.

Every agent has the same small interface:

* ``choose(board)`` returns one of ``board.MOVES`` (or ``None`` if no move is legal)
* ``nodes`` holds how many search nodes the last decision visited

Search agents:

* :class:`ExpectimaxAgent` - max nodes for the player, chance nodes average over
  every possible tile spawn (2 with p=0.9, 4 with p=0.1). Branches whose
  probability drops below a threshold are cut, and chance nodes are memoised.
* :class:`MinimaxAgent` - treats the tile spawn as an opponent that always picks
  the worst cell and value. ``pruning=True`` adds alpha-beta pruning;
  ``pruning=False`` is the plain version, kept to show how many nodes pruning saves.
* :class:`MCTSAgent` - open-loop Monte Carlo Tree Search (UCT). The tree stores
  move sequences; random spawns are re-sampled on every iteration.

Baselines: :class:`RandomAgent` and :class:`GreedyAgent` (one-ply heuristic).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from .board import MOVES, Board, count_empty, empty_cells, legal_moves, move, move_with_score, set_rank, spawn
from .heuristics import evaluate

INF = float("inf")


class Agent:
    name = "agent"
    label = "Agent"

    def __init__(self) -> None:
        self.nodes = 0

    def choose(self, board: Board) -> int | None:  # pragma: no cover - interface
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"
    label = "Random"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> int | None:
        moves = legal_moves(board)
        self.nodes = len(moves)
        return self.rng.choice(moves) if moves else None


class GreedyAgent(Agent):
    """Pick the move whose resulting board scores best, without looking further."""

    name = "greedy"
    label = "Greedy"

    def choose(self, board: Board) -> int | None:
        self.nodes = 0
        best_move, best_value = None, -INF
        for m in MOVES:
            after = move(board, m)
            if after == board:
                continue
            self.nodes += 1
            value = evaluate(after)
            if value > best_value:
                best_move, best_value = m, value
        return best_move


class ExpectimaxAgent(Agent):
    name = "expectimax"
    label = "Expectimax"

    def __init__(self, depth: int = 2, prob_cutoff: float = 1e-4, adaptive: bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.prob_cutoff = prob_cutoff
        self.adaptive = adaptive
        self._depth_limit = depth
        self._cache: dict[Board, tuple] = {}

    def choose(self, board: Board) -> int | None:
        self.nodes = 0
        self._cache = {}
        self._depth_limit = self.depth
        if self.adaptive and count_empty(board) <= 3:
            # Few empty cells: the position is dangerous and the tree is small,
            # so spend one extra level there.
            self._depth_limit += 1

        best_move, best_value = None, -INF
        for m in MOVES:
            after = move(board, m)
            if after == board:
                continue
            value = self._chance(after, 1.0, 1)
            if value > best_value:
                best_move, best_value = m, value
        return best_move

    def _chance(self, board: Board, prob: float, depth: int) -> float:
        if depth > self._depth_limit or prob < self.prob_cutoff:
            return evaluate(board)
        cached = self._cache.get(board)
        if cached is not None and cached[0] <= depth:
            return cached[1]

        cells = empty_cells(board)
        share = prob / len(cells)
        total = 0.0
        for i in cells:
            total += 0.9 * self._max(set_rank(board, i, 1), share * 0.9, depth)
            total += 0.1 * self._max(set_rank(board, i, 2), share * 0.1, depth)
        value = total / len(cells)
        self._cache[board] = (depth, value)
        return value

    def _max(self, board: Board, prob: float, depth: int) -> float:
        self.nodes += 1
        best = 0.0  # no legal move = lost, the worst possible value
        for m in MOVES:
            after = move(board, m)
            if after != board:
                value = self._chance(after, prob, depth + 1)
                if value > best:
                    best = value
        return best


class MinimaxAgent(Agent):
    name = "minimax"
    label = "Minimax"

    def __init__(self, depth: int = 2, pruning: bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.pruning = pruning
        self.value = 0.0  # minimax value of the last decision
        if not pruning:
            self.name = "minimax_np"
            self.label = "Minimax (no pruning)"

    def choose(self, board: Board) -> int | None:
        self.nodes = 0
        best_move, alpha = None, -INF
        for m in self._ordered_moves(board):
            after = move(board, m)
            value = self._min(after, self.depth, alpha, INF)
            if value > alpha or best_move is None:
                best_move, alpha = m, max(alpha, value)
        self.value = alpha
        return best_move

    def _ordered_moves(self, board: Board) -> list[int]:
        """Legal moves, most promising first (better ordering = more cut-offs)."""
        options = [(evaluate(move(board, m)), m) for m in MOVES if move(board, m) != board]
        options.sort(reverse=True)
        return [m for _, m in options]

    def _max(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if depth == 0:
            return evaluate(board)
        best = -INF
        moves = self._ordered_moves(board) if self.pruning else legal_moves(board)
        for m in moves:
            best = max(best, self._min(move(board, m), depth, alpha, beta))
            if self.pruning:
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        return best if best > -INF else 0.0  # no legal move = lost

    def _min(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        best = INF
        for i in empty_cells(board):
            for rank in (1, 2):
                best = min(best, self._max(set_rank(board, i, rank), depth - 1, alpha, beta))
                if self.pruning:
                    beta = min(beta, best)
                    if alpha >= beta:
                        return best
        return best


class _TreeNode:
    __slots__ = ("children", "visits", "total")

    def __init__(self) -> None:
        self.children: dict[int, _TreeNode] = {}
        self.visits = 0
        self.total = 0.0


class MCTSAgent(Agent):
    name = "mcts"
    label = "MCTS"

    def __init__(
        self, iterations: int = 300, rollout_moves: int = 20, exploration: float = 1.0, seed: int | None = None
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.rollout_moves = rollout_moves
        self.exploration = exploration
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> int | None:
        root_moves = legal_moves(board)
        self.nodes = 0
        if len(root_moves) <= 1:
            return root_moves[0] if root_moves else None

        rng = self.rng
        root = _TreeNode()
        best_reward = 1.0

        for _ in range(self.iterations):
            state = board
            node = root
            path = [root]
            reward = 0.0

            # Selection + expansion: walk down the tree until we add one new node.
            while True:
                moves = legal_moves(state)
                if not moves:
                    break
                untried = [m for m in moves if m not in node.children]
                if untried:
                    m = rng.choice(untried)
                    node.children[m] = child = _TreeNode()
                else:
                    log_n = math.log(node.visits)
                    m = max(moves, key=lambda mv: self._ucb(node.children[mv], log_n, best_reward))
                    child = node.children[m]
                state, gained = move_with_score(state, m)
                state = spawn(state, rng)
                reward += gained
                path.append(child)
                node = child
                self.nodes += 1
                if untried:
                    break

            # Rollout: random moves from the new node.
            for _ in range(self.rollout_moves):
                moves = legal_moves(state)
                if not moves:
                    break
                state, gained = move_with_score(state, rng.choice(moves))
                state = spawn(state, rng)
                reward += gained
            if not legal_moves(state):
                reward *= 0.5  # dying inside the rollout is bad news

            best_reward = max(best_reward, reward)
            for n in path:
                n.visits += 1
                n.total += reward

        return max(root.children.items(), key=lambda item: item[1].visits)[0]

    def _ucb(self, child: _TreeNode, log_parent_visits: float, scale: float) -> float:
        mean = child.total / child.visits / scale
        return mean + self.exploration * math.sqrt(log_parent_visits / child.visits)


AGENTS: dict[str, Callable[[], Agent]] = {
    "expectimax": ExpectimaxAgent,
    "minimax": MinimaxAgent,
    "minimax_np": lambda: MinimaxAgent(pruning=False),
    "mcts": MCTSAgent,
    "greedy": GreedyAgent,
    "random": RandomAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {', '.join(AGENTS)}") from None
