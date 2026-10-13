import random

import pytest

from solver2048.board import (
    DOWN,
    LEFT,
    MOVES,
    RIGHT,
    UP,
    count_empty,
    empty_cells,
    from_grid,
    is_game_over,
    legal_moves,
    max_tile,
    move,
    move_with_score,
    new_game,
    spawn,
    to_grid,
    transpose,
)


def reference_slide(line):
    """Obvious, slow implementation of one row sliding left."""
    tiles = [v for v in line if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            score += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (4 - len(out)), score


def reference_move(grid, direction):
    g = [row[:] for row in grid]
    score = 0
    if direction in (LEFT, RIGHT):
        for r in range(4):
            line = g[r] if direction == LEFT else g[r][::-1]
            new, gained = reference_slide(line)
            g[r] = new if direction == LEFT else new[::-1]
            score += gained
    else:
        for c in range(4):
            col = [g[r][c] for r in range(4)]
            line = col if direction == UP else col[::-1]
            new, gained = reference_slide(line)
            new = new if direction == UP else new[::-1]
            for r in range(4):
                g[r][c] = new[r]
            score += gained
    return g, score


def random_grid(rng, fill=0.6, top=11):
    return [[(1 << rng.randint(1, top)) if rng.random() < fill else 0 for _ in range(4)] for _ in range(4)]


@pytest.mark.parametrize(
    "row, expected, score",
    [
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),  # the new 4 must not merge again in the same move
        ([4, 4, 4, 0], [8, 4, 0, 0], 8),
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),
        ([2, 0, 2, 0], [4, 0, 0, 0], 4),
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),
        ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
    ],
)
def test_left_slide_rules(row, expected, score):
    board = from_grid([row, [0] * 4, [0] * 4, [0] * 4])
    after, gained = move_with_score(board, LEFT)
    assert to_grid(after)[0] == expected
    assert gained == score


def test_grid_round_trip():
    rng = random.Random(1)
    for _ in range(200):
        grid = random_grid(rng)
        assert to_grid(from_grid(grid)) == grid


def test_transpose_matches_definition_and_is_an_involution():
    rng = random.Random(2)
    for _ in range(200):
        grid = random_grid(rng)
        board = from_grid(grid)
        assert to_grid(transpose(board)) == [list(col) for col in zip(*grid, strict=False)]
        assert transpose(transpose(board)) == board


def test_every_direction_matches_the_reference_on_random_boards():
    rng = random.Random(3)
    for _ in range(2000):
        grid = random_grid(rng, fill=rng.random())
        board = from_grid(grid)
        for direction in MOVES:
            expected_grid, expected_score = reference_move(grid, direction)
            after, gained = move_with_score(board, direction)
            assert to_grid(after) == expected_grid
            if after != board:
                assert gained == expected_score


def test_empty_cells_and_count_agree():
    rng = random.Random(4)
    for _ in range(500):
        grid = random_grid(rng, fill=rng.random())
        board = from_grid(grid)
        expected = sum(v == 0 for row in grid for v in row)
        assert count_empty(board) == expected == len(empty_cells(board))


def test_spawn_adds_one_small_tile():
    rng = random.Random(5)
    board = new_game(rng)
    tiles = [v for row in to_grid(board) for v in row if v]
    assert len(tiles) == 2 and set(tiles) <= {2, 4}
    after = spawn(board, rng)
    assert count_empty(after) == count_empty(board) - 1


def test_game_over_detection():
    stuck = from_grid([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert is_game_over(stuck)
    assert legal_moves(stuck) == []
    alive = from_grid([[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]])
    assert not is_game_over(alive)
    assert set(legal_moves(alive)) == {LEFT, RIGHT}


def test_max_tile():
    assert max_tile(0) == 0
    assert max_tile(from_grid([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2048, 0], [0, 0, 0, 4]])) == 2048


def test_invalid_tiles_are_rejected():
    with pytest.raises(ValueError):
        from_grid([[3, 0, 0, 0]] + [[0] * 4] * 3)


def test_unchanged_board_means_illegal_move():
    board = from_grid([[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4])
    assert move(board, LEFT) == board
    assert move(board, RIGHT) == board
    assert move(board, DOWN) != board
