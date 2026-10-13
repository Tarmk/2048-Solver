"""Headless benchmark: let agents play full games and compare them.

    python -m solver2048.bench                       # every agent, 10 games each
    python -m solver2048.bench -a expectimax -n 50   # one agent, more games
    python -m solver2048.bench --markdown            # print a Markdown table

Games use fixed seeds (``--seed`` + game number), so every agent sees the same
tile spawns at the start and runs are reproducible.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from .agents import AGENTS, make_agent
from .game import GameResult, play_game


def _run_one(job: tuple) -> GameResult:
    name, seed, max_moves, stop_at = job
    return play_game(make_agent(name), seed=seed, max_moves=max_moves, stop_at=stop_at)


def summarise(results: Sequence[GameResult]) -> dict:
    tiles = Counter(r.max_tile for r in results)
    moves = sum(r.moves for r in results)
    return {
        "agent": results[0].agent,
        "games": len(results),
        "win_rate": sum(r.max_tile >= 2048 for r in results) / len(results),
        "median_score": statistics.median(r.score for r in results),
        "best_tile": max(r.max_tile for r in results),
        "tiles": dict(sorted(tiles.items(), reverse=True)),
        "ms_per_move": 1000.0 * sum(r.seconds for r in results) / moves if moves else 0.0,
        "nodes_per_move": sum(r.nodes for r in results) / moves if moves else 0.0,
    }


def format_table(rows: list[dict], markdown: bool) -> str:
    header = ["Agent", "Games", "Reached 2048", "Median score", "Best tile", "ms / move", "Nodes / move"]
    lines = []
    for row in rows:
        lines.append(
            [
                row["agent"],
                str(row["games"]),
                f"{row['win_rate']:.0%}",
                f"{row['median_score']:,.0f}",
                str(row["best_tile"]),
                f"{row['ms_per_move']:.2f}" if row["ms_per_move"] < 10 else f"{row['ms_per_move']:.1f}",
                f"{row['nodes_per_move']:,.0f}",
            ]
        )
    if markdown:
        out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(line) + " |" for line in lines]
        return "\n".join(out)
    widths = [max(len(h), *(len(line[i]) for line in lines)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(
        [fmt.format(*header), fmt.format(*("-" * w for w in widths))] + [fmt.format(*line) for line in lines]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the 2048 agents headlessly.")
    parser.add_argument(
        "-a", "--agent", action="append", choices=list(AGENTS), help="agent to run (repeatable); default: all"
    )
    parser.add_argument("-n", "--games", type=int, default=10, help="games per agent (default 10)")
    parser.add_argument("--seed", type=int, default=0, help="first seed (default 0)")
    parser.add_argument("--max-moves", type=int, default=None, help="cut games off after this many moves")
    parser.add_argument("--stop-at", type=int, default=None, help="end a game once this tile appears")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="games to run in parallel (default 1)")
    parser.add_argument("-q", "--quiet", action="store_true", help="no per-game lines")
    args = parser.parse_args(argv)

    rows = []
    pool: ProcessPoolExecutor | None = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    try:
        for name in args.agent or list(AGENTS):
            jobs = [(name, args.seed + i, args.max_moves, args.stop_at) for i in range(args.games)]
            outcomes = pool.map(_run_one, jobs) if pool else map(_run_one, jobs)
            results = []
            for i, result in enumerate(outcomes):
                results.append(result)
                if not args.quiet:
                    print(
                        f"{name:<11} game {i + 1:>3}: score {result.score:>7,}  max tile {result.max_tile:>5}  "
                        f"{result.moves:>5} moves  {result.ms_per_move:6.1f} ms/move",
                        file=sys.stderr,
                    )
            rows.append(summarise(results))
    finally:
        if pool:
            pool.shutdown()

    print(format_table(rows, args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
