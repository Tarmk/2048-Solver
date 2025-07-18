# 2048 Solver with AI Agents (Pygame)

<div align="center">
  
**Play manually … or let four different AI agents battle it out for you!**  

</div>

---

## Table of Contents
1. [Features](#features)
2. [Demo Video / Screenshot](#demo)
3. [Installation](#installation)
4. [Running the Game](#running-the-game)
5. [Controls](#controls)
6. [AI Agents](#ai-agents)
7. [Live Metrics Panel](#live-metrics-panel)
8. [Tuning the Difficulty](#tuning-the-difficulty)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License & Acknowledgements](#license--acknowledgements)

---

## Features<a name="features"></a>
- 🎮 **Full 2048 gameplay** with smooth animations and polished UI (rounded tiles, drop shadows, etc.).
- 🤖 **Four built-in AI agents** (Minimax, Expectimax, Monte-Carlo Tree Search, and Naïve Minimax) that can play the game for you.
- 📊 **Real-time statistics panel** showing nodes searched, decision time and a win checkbox for each algorithm.
- ⚡ **Batch mode** to benchmark all agents sequentially without user interaction.
- 🖱️ **Clickable buttons** to toggle each agent or start/stop the batch run directly from the GUI.
- 🛠 **Parameter sliders** *(coming soon)* – quickly adjust depths / simulation counts without editing code.

## Demo<a name="demo"></a>
*A demo GIF/Screenshot can be placed here.*  
If you record a short GIF, save it as `docs/demo.gif` and embed it like:
```md
![Game demo](docs/demo.gif)
```
*(Remote images are stripped out in some environments, so keep it local.)*

---

## Installation<a name="installation"></a>
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your_user>/2048-Solver.git
   cd 2048-Solver
   ```
2. **Create a virtual environment** *(optional but recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # or simply: pip install pygame
   ```
   Only `pygame` (≥ 2.0) is required.  The AI code uses standard-library modules only.

---

## Running the Game<a name="running-the-game"></a>
```bash
python 2048.py
```
A window opens with a 4×4 grid just like the original 2048 game.

---

## Controls<a name="controls"></a>
### Human
| Key | Action |
|-----|--------|
| ← ↑ ↓ → | Move tiles |
| **R** | Reset the current board |

### AI & Utility
| Key | Action |
|-----|--------|
| **M** | Toggle **Minimax** agent |
| **E** | Toggle **Expectimax** agent |
| **N** | Toggle **MCTS** agent |
| **B** | Toggle **Naïve Minimax** agent |
| **A** | Start / stop **batch mode** (runs all agents sequentially) |

*(The same actions are available through on-screen buttons.)*

---

## AI Agents<a name="ai-agents"></a>
| Agent | Strategy | Adjustable Parameter |
|-------|----------|----------------------|
| **Minimax** | Two-player adversarial search with alpha–beta pruning | `self.minimax_depth` |
| **Expectimax** | Chance nodes model random tile spawns | `self.expectimax_depth` |
| **MCTS** | Monte-Carlo Tree Search with UCT | `self.base_mcts_simulations` |
| **Naïve Minimax** | Same as Minimax but **without pruning** (for comparison) | `self.minimax_depth` |

> All parameters live in `Game2048.__init__` – tweak them and instantly see the effect on playing strength and speed.

---

## Live Metrics Panel<a name="live-metrics-panel"></a>
A sidebar tracks for each agent:
- **Nodes** searched so far
- **Cumulative decision time** (seconds)
- **Win checkbox** – ticks itself automatically when the agent ever reaches 2048 in the current session (you can also toggle it manually).

---

## Tuning the Difficulty<a name="tuning-the-difficulty"></a>
Increasing the search depth or the number of simulations will make an agent stronger **but** also more CPU-intensive.

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `self.minimax_depth` | 3-6 | +1 depth ≈ 4× nodes |
| `self.expectimax_depth` | 3-5 | Similar to Minimax |
| `self.base_mcts_simulations` | 50-400 | Linear in simulations |

Experiment and find a good balance for your machine.

---

## Troubleshooting<a name="troubleshooting"></a>
| Problem | Fix |
|---------|-----|
| *`ModuleNotFoundError: No module named 'pygame'`* | Make sure you ran `pip install pygame` inside the activated environment. |
| Pygame window is blank / freezes on macOS Sonoma | Ensure you are using Python 3.11+ and Pygame ≥ 2.4.  If the problem persists try `brew install sdl2` and reinstall `pygame`. |
| High CPU usage | Lower the search depth or the number of simulations. |

---

## Contributing<a name="contributing"></a>
Pull requests are very welcome!  Feel free to open an issue to discuss bugs, feature ideas or algorithm tweaks.

1. Fork 🍴, clone, create a branch.
2. Make your changes (please keep each PR focused).
3. Run `python -m flake8` – we aim for a clean lint.
4. Submit PR and describe *what* and *why*.

---

## License & Acknowledgements<a name="license--acknowledgements"></a>
This project is **MIT-licensed**.  Original 2048 concept by **[Gabriele Cirulli](https://gabrielecirulli.com/)**.  UI style inspired by the open-source community.