import random
import statistics

import pytest

from solver2048.agents import AGENTS, ExpectimaxAgent, MinimaxAgent, RandomAgent, make_agent
from solver2048.board import from_grid, legal_moves, new_game
from solver2048.game import Game, play_game
from solver2048.heuristics import evaluate

MID_GAME = from_grid([[2, 4, 8, 16], [0, 2, 4, 32], [0, 0, 2, 64], [0, 0, 0, 128]])
STUCK = from_grid([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])


@pytest.mark.parametrize("name", list(AGENTS))
def test_agents_return_a_legal_move(name):
    agent = make_agent(name)
    rng = random.Random(7)
    board = new_game(rng)
    for b in (board, MID_GAME):
        assert agent.choose(b) in legal_moves(b)
        assert agent.nodes >= 1


@pytest.mark.parametrize("name", list(AGENTS))
def test_agents_return_none_when_stuck(name):
    assert make_agent(name).choose(STUCK) is None


def test_unknown_agent_name():
    with pytest.raises(ValueError):
        make_agent("nope")


def test_alpha_beta_gives_the_same_value_with_fewer_nodes():
    rng = random.Random(11)
    pruned, naive = MinimaxAgent(depth=2), MinimaxAgent(depth=2, pruning=False)
    game = Game(seed=3, board=MID_GAME)
    for _ in range(8):
        a = pruned.choose(game.board)
        naive.choose(game.board)
        assert pruned.value == pytest.approx(naive.value)
        assert pruned.nodes < naive.nodes
        game.play(a)
        game.play(rng.choice(legal_moves(game.board)))


def test_heuristic_prefers_a_tidy_corner():
    tidy = from_grid([[256, 128, 64, 32], [0, 0, 0, 16], [0, 0, 0, 0], [0, 0, 0, 0]])
    messy = from_grid([[32, 0, 256, 0], [0, 128, 0, 16], [64, 0, 0, 0], [0, 0, 0, 0]])
    assert evaluate(tidy) > evaluate(messy)


def test_expectimax_beats_random():
    """Short games (200 moves) so the test stays fast; expectimax should score far higher."""
    seeds = range(3)
    smart = [play_game(ExpectimaxAgent(depth=1), seed=s, max_moves=200).score for s in seeds]
    dumb = [play_game(RandomAgent(seed=s), seed=s, max_moves=200).score for s in seeds]
    assert statistics.mean(smart) > 1.5 * statistics.mean(dumb)


def test_games_are_reproducible_with_a_seed():
    a = play_game(ExpectimaxAgent(depth=1), seed=42, max_moves=60)
    b = play_game(ExpectimaxAgent(depth=1), seed=42, max_moves=60)
    assert (a.score, a.moves, a.max_tile) == (b.score, b.moves, b.max_tile)


def test_game_rejects_illegal_moves():
    game = Game(seed=1, board=from_grid([[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4]))
    before = (game.board, game.score, game.moves)
    from solver2048.board import LEFT

    assert game.play(LEFT) is False
    assert (game.board, game.score, game.moves) == before
