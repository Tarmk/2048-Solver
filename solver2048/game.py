"""A game of 2048 without any UI: board, score and move counting."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .agents import Agent
from .board import Board, is_game_over, max_tile, move_with_score, new_game, spawn


@dataclass
class Game:
    seed: int | None = None
    board: Board = 0
    score: int = 0
    moves: int = 0
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.board:
            self.board = new_game(self.rng)

    @property
    def max_tile(self) -> int:
        return max_tile(self.board)

    @property
    def over(self) -> bool:
        return is_game_over(self.board)

    def play(self, direction: int) -> bool:
        """Apply a move and spawn a tile. Returns False if the move was illegal."""
        after, gained = move_with_score(self.board, direction)
        if after == self.board:
            return False
        self.board = spawn(after, self.rng)
        self.score += gained
        self.moves += 1
        return True


@dataclass
class GameResult:
    agent: str
    seed: int | None
    score: int
    max_tile: int
    moves: int
    seconds: float
    nodes: int

    @property
    def ms_per_move(self) -> float:
        return 1000.0 * self.seconds / self.moves if self.moves else 0.0


def play_game(
    agent: Agent, seed: int | None = None, max_moves: int | None = None, stop_at: int | None = None
) -> GameResult:
    """Let ``agent`` play one full game. ``stop_at`` ends the game once that tile appears."""
    game = Game(seed=seed)
    nodes = 0
    thinking = 0.0
    while not game.over:
        if max_moves is not None and game.moves >= max_moves:
            break
        if stop_at is not None and game.max_tile >= stop_at:
            break
        start = time.perf_counter()
        choice = agent.choose(game.board)
        thinking += time.perf_counter() - start
        nodes += agent.nodes
        if choice is None or not game.play(choice):
            break
    return GameResult(agent.name, seed, game.score, game.max_tile, game.moves, thinking, nodes)
