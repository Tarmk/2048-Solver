"""Board evaluation used by the search agents.

The scoring is the row heuristic popularised by Robert Xiao's (nneonneo) 2048 AI:
reward empty cells and possible merges, punish rows and columns that are not
monotonic, and punish large tile sums slightly so the search prefers merging.
Like the moves, it is precomputed for every 16-bit row, so evaluating a board
is eight table lookups (four rows plus four columns).
"""

from __future__ import annotations

from .board import ROW_MASK, Board, transpose

LOST_PENALTY = 200_000.0
MONOTONICITY_POWER = 4.0
MONOTONICITY_WEIGHT = 47.0
SUM_POWER = 3.5
SUM_WEIGHT = 11.0
MERGES_WEIGHT = 700.0
EMPTY_WEIGHT = 270.0


def _row_heuristic(row: int) -> float:
    line = [(row >> (4 * i)) & 0xF for i in range(4)]

    total = 0.0
    empty = 0
    merges = 0
    prev = 0
    run = 0
    for rank in line:
        total += rank**SUM_POWER
        if rank == 0:
            empty += 1
            continue
        if prev == rank:
            run += 1
        elif run > 0:
            merges += 1 + run
            run = 0
        prev = rank
    if run > 0:
        merges += 1 + run

    mono_left = 0.0
    mono_right = 0.0
    for i in range(1, 4):
        a = line[i - 1] ** MONOTONICITY_POWER
        b = line[i] ** MONOTONICITY_POWER
        if line[i - 1] > line[i]:
            mono_left += a - b
        else:
            mono_right += b - a

    return (
        LOST_PENALTY
        + EMPTY_WEIGHT * empty
        + MERGES_WEIGHT * merges
        - MONOTONICITY_WEIGHT * min(mono_left, mono_right)
        - SUM_WEIGHT * total
    )


_ROW_HEURISTIC = [_row_heuristic(row) for row in range(65536)]


def _rows_value(b: Board) -> float:
    h = _ROW_HEURISTIC
    return h[b & ROW_MASK] + h[(b >> 16) & ROW_MASK] + h[(b >> 32) & ROW_MASK] + h[(b >> 48) & ROW_MASK]


def evaluate(b: Board) -> float:
    """Higher is better. Rows and columns are scored the same way."""
    return _rows_value(b) + _rows_value(transpose(b))
