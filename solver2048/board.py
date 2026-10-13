"""Bitboard engine for 2048.

The whole 4x4 board is one 64-bit integer. Each cell is a 4-bit "rank":
0 means empty, otherwise the tile value is ``2 ** rank`` (so 1 -> 2, 11 -> 2048).
Cell (row, col) lives in bits ``4 * (4 * row + col)``.

Every possible 16-bit row is precomputed once at import time (65,536 entries),
so a move is four table lookups instead of a Python loop over cells. Up and down
reuse the left/right tables on the transposed board.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

Board = int

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = (UP, DOWN, LEFT, RIGHT)
MOVE_NAMES = ("up", "down", "left", "right")

ROW_MASK = 0xFFFF
COL_MASK = 0x000F_000F_000F_000F
MAX_RANK = 15  # a nibble cannot hold more than 2**15 = 32768


def _row_cells(row: int) -> list[int]:
    return [(row >> (4 * i)) & 0xF for i in range(4)]


def _cells_row(cells: Sequence[int]) -> int:
    return cells[0] | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)


def _reverse_row(row: int) -> int:
    return ((row >> 12) & 0xF) | ((row >> 4) & 0xF0) | ((row << 4) & 0xF00) | ((row << 12) & 0xF000)


def _slide_left(cells: Sequence[int]) -> tuple[list[int], int]:
    """Slide one row towards index 0. Each tile merges at most once per move."""
    tiles = [c for c in cells if c]
    out: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_RANK:
            rank = tiles[i] + 1
            out.append(rank)
            score += 1 << rank
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (4 - len(out)))
    return out, score


def _build_tables():
    left = [0] * 65536
    right = [0] * 65536
    score = [0] * 65536
    for row in range(65536):
        cells, gained = _slide_left(_row_cells(row))
        left[row] = _cells_row(cells)
        score[row] = gained
    for row in range(65536):
        rev = _reverse_row(row)
        right[row] = _reverse_row(left[rev])
    return left, right, score


_LEFT, _RIGHT, _SCORE = _build_tables()


def transpose(b: Board) -> Board:
    """Swap rows and columns (cell (r, c) <-> (c, r)) with bit tricks."""
    a1 = b & 0xF0F0_0F0F_F0F0_0F0F
    a2 = b & 0x0000_F0F0_0000_F0F0
    a3 = b & 0x0F0F_0000_0F0F_0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00_FF00_00FF_00FF
    b2 = a & 0x00FF_00FF_0000_0000
    b3 = a & 0x0000_0000_FF00_FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _apply_rows(b: Board, table: list[int]) -> Board:
    return (
        table[b & ROW_MASK]
        | (table[(b >> 16) & ROW_MASK] << 16)
        | (table[(b >> 32) & ROW_MASK] << 32)
        | (table[(b >> 48) & ROW_MASK] << 48)
    )


def _row_score(b: Board) -> int:
    return (
        _SCORE[b & ROW_MASK]
        + _SCORE[(b >> 16) & ROW_MASK]
        + _SCORE[(b >> 32) & ROW_MASK]
        + _SCORE[(b >> 48) & ROW_MASK]
    )


def move(b: Board, direction: int) -> Board:
    """Return the board after ``direction``. Unchanged board means the move is illegal."""
    if direction == LEFT:
        return _apply_rows(b, _LEFT)
    if direction == RIGHT:
        return _apply_rows(b, _RIGHT)
    t = transpose(b)
    if direction == UP:
        return transpose(_apply_rows(t, _LEFT))
    if direction == DOWN:
        return transpose(_apply_rows(t, _RIGHT))
    raise ValueError(f"unknown direction {direction!r}")


def move_with_score(b: Board, direction: int) -> tuple[Board, int]:
    """Like :func:`move` but also return the points gained from merges."""
    if direction in (LEFT, RIGHT):
        # Merges in a row happen between the same pairs whichever way it slides.
        return move(b, direction), _row_score(b)
    return move(b, direction), _row_score(transpose(b))


def legal_moves(b: Board) -> list[int]:
    return [m for m in MOVES if move(b, m) != b]


def is_game_over(b: Board) -> bool:
    return not legal_moves(b)


def count_empty(b: Board) -> int:
    x = b | ((b >> 2) & 0x3333_3333_3333_3333)
    x |= x >> 1
    return (~x & 0x1111_1111_1111_1111).bit_count()


def empty_cells(b: Board) -> list[int]:
    """Cell indices (0..15, row-major) that are empty."""
    return [i for i in range(16) if not (b >> (4 * i)) & 0xF]


def rank_at(b: Board, row: int, col: int) -> int:
    return (b >> (4 * (4 * row + col))) & 0xF


def set_rank(b: Board, index: int, rank: int) -> Board:
    shift = 4 * index
    return (b & ~(0xF << shift)) | (rank << shift)


def max_rank(b: Board) -> int:
    return max((b >> (4 * i)) & 0xF for i in range(16))


def max_tile(b: Board) -> int:
    rank = max_rank(b)
    return 1 << rank if rank else 0


def spawn(b: Board, rng: random.Random | None = None) -> Board:
    """Drop a 2 (90%) or a 4 (10%) on a random empty cell, like the real game."""
    rng = rng or random
    cells = empty_cells(b)
    if not cells:
        return b
    rank = 1 if rng.random() < 0.9 else 2
    return set_rank(b, rng.choice(cells), rank)


def new_game(rng: random.Random | None = None) -> Board:
    return spawn(spawn(0, rng), rng)


def from_grid(grid: Sequence[Sequence[int]]) -> Board:
    """Build a board from a 4x4 list of tile values (0 for empty)."""
    b = 0
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value:
                rank = value.bit_length() - 1
                if 1 << rank != value or not 1 <= rank <= MAX_RANK:
                    raise ValueError(f"not a valid tile value: {value}")
                b = set_rank(b, 4 * r + c, rank)
    return b


def to_grid(b: Board) -> list[list[int]]:
    """Turn a board back into a 4x4 list of tile values."""
    grid = []
    for r in range(4):
        row = []
        for c in range(4):
            rank = rank_at(b, r, c)
            row.append(1 << rank if rank else 0)
        grid.append(row)
    return grid


def rows(b: Board) -> Iterator[int]:
    for r in range(4):
        yield (b >> (16 * r)) & ROW_MASK


def format_board(b: Board) -> str:
    return "\n".join(" ".join(f"{v:>5}" if v else "    ." for v in row) for row in to_grid(b))
