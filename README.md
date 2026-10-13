# 2048 Solver

[![CI](https://github.com/Tarmk/2048-Solver/actions/workflows/ci.yml/badge.svg)](https://github.com/Tarmk/2048-Solver/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-3776ab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Play 2048 yourself, or watch four search agents play it: **Expectimax**, **Minimax with alpha-beta pruning**, **plain Minimax** (to show what pruning saves) and **Monte Carlo Tree Search**. A headless benchmark plays full games and compares them.

![Expectimax mid-game, with the per-agent stats table on the right](docs/demo.png)

## Results

Every agent played 20 full games (seeds 0 to 19, so all agents get the same starting spawns) on my desktop: AMD Zen 5 CPU, Python 3.14, eight games at a time.

| Agent | Reached 2048 | Median score | Highest tile reached (games) | Time per move | Nodes per move |
|---|---|---|---|---|---|
| **Expectimax** | **20 / 20** | **133,922** | 16384 (1), 8192 (13), 4096 (5), 2048 (1) | 27 ms | 3,581 |
| MCTS | 18 / 20 | 36,138 | 4096 (2), 2048 (16), 1024 (2) | 52 ms | 1,159 |
| Minimax | 10 / 20 | 18,966 | 2048 (10), 1024 (6), 512 (3), 256 (1) | 0.6 ms | 267 |
| Minimax, no pruning | 10 / 20 | 18,966 | same games as Minimax | 2.5 ms | 1,625 |
| Greedy (one ply) | 0 / 20 | 4,564 | 512 (9), 256 (11) | <0.1 ms | 4 |
| Random | 0 / 20 | 1,046 | 256 (1), 128 (10), 64 (8), 16 (1) | <0.1 ms | 3 |

- **Expectimax is the strongest** because it models the spawn honestly: tiles appear at random, not where they hurt most. It made 2048 in every game and 8192 in 14 of 20.
- **Minimax plays too defensively.** It assumes the worst tile always lands in the worst cell, which is not how 2048 works.
- **Alpha-beta pruning is free strength.** With and without pruning, Minimax played identical games, but pruning needed about 6× fewer nodes per move.

Reproduce with `python -m solver2048.bench -n 20 -j 8 --markdown`.

## How it works

**Bitboard engine** ([`solver2048/board.py`](solver2048/board.py)). The board is a single 64-bit integer, four bits per cell, holding the tile's exponent (`11` means 2048). All 65,536 possible rows are slid once at import time and stored in lookup tables, so a move is four table lookups. Up and down transpose the board with a few bit operations and reuse the left/right tables. The engine is tested against a plain list-based implementation on thousands of random boards.

**Evaluation** ([`solver2048/heuristics.py`](solver2048/heuristics.py)). Rewards empty cells and adjacent equal tiles, penalises rows and columns that are not monotonic, and lightly penalises the tile sum so merging is preferred. The weights come from [Robert Xiao's 2048 AI](https://github.com/nneonneo/2048-ai). The heuristic is also precomputed per row, so scoring a board is eight lookups.

**Agents** ([`solver2048/agents.py`](solver2048/agents.py)):

| Agent | Idea | Default setting |
|---|---|---|
| Expectimax | Max nodes for the player. Chance nodes average over every empty cell and both spawn values (2 at 90%, 4 at 10%). Branches less likely than 1 in 10,000 are cut, and chance nodes are cached within a move. | 2 plies, 3 when 3 or fewer cells are empty |
| Minimax | Treats the tile spawn as an opponent that always places the worst tile. Moves are ordered by the heuristic so alpha-beta cuts early. | 2 plies |
| Minimax, no pruning | Same search without alpha-beta: same decisions, about 6× the nodes. | 2 plies |
| MCTS | Open-loop UCT: the tree stores move sequences, and random spawns are re-sampled on every iteration. Rollouts are random. | 300 iterations, 20-move rollouts |
| Greedy, Random | Baselines for the benchmark (one-ply heuristic and uniform random). | |

In the Pygame app the search runs on a worker thread, so the window stays responsive while an agent thinks.

## Running it

Python 3.10 or newer.

```bash
git clone https://github.com/Tarmk/2048-Solver.git
cd 2048-Solver
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # only pygame-ce
python -m solver2048             # or: python 2048.py
```

`python -m solver2048 --agent expectimax --seed 1` starts with an agent already playing.

### Controls

| Key | Action |
|---|---|
| Arrow keys / WASD | Move (when no agent is playing) |
| E / M / B / N | Start or stop Expectimax / Minimax / Minimax without pruning / MCTS |
| Tab | Batch run: each agent plays one game until it makes 2048 |
| + / - | Faster / slower agent moves |
| R | New game |
| Esc | Quit |

Buttons on the right do the same as the keys. The table under them tracks moves, nodes searched per move, thinking time per move and best tile for each agent during the session.

### Benchmark

```bash
python -m solver2048.bench                         # every agent, 10 games each
python -m solver2048.bench -a expectimax -n 50 -j 4  # one agent, 50 games, 4 processes
python -m solver2048.bench --markdown              # Markdown table, as above
```

Game *i* uses seed *i*, so every agent gets the same random spawns and runs are reproducible.

## Project layout

```
solver2048/
  board.py       bitboard engine: moves, spawns, conversions
  heuristics.py  board evaluation (precomputed per row)
  agents.py      Expectimax, Minimax, MCTS, Greedy, Random
  game.py        headless game loop shared by the UI and the benchmark
  bench.py       benchmark CLI
  ui.py          Pygame front-end
tests/           pytest: engine against a reference implementation, agent behaviour
2048.py          launcher kept for `python 2048.py`
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

CI runs the tests on Python 3.10, 3.12 and 3.13, a short benchmark, and a headless run of the Pygame app.

## License

MIT, see [LICENSE](LICENSE). 2048 was created by [Gabriele Cirulli](https://github.com/gabrielecirulli/2048).
