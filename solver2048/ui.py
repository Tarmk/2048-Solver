"""Pygame front-end: play 2048 yourself or watch the agents play.

Run with ``python -m solver2048`` (or ``python 2048.py``).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import pygame

from .agents import Agent, make_agent
from .board import DOWN, LEFT, RIGHT, UP, max_tile, to_grid
from .game import Game

WINDOW_W, WINDOW_H = 1060, 700
CELL, GAP = 110, 12
GRID_X, GRID_Y = 40, 160
GRID_SIZE = 4 * CELL + 5 * GAP
PANEL_X = GRID_X + GRID_SIZE + 40

BACKGROUND = (250, 248, 239)
BOARD_BG = (187, 173, 160)
EMPTY = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)
BUTTON = (187, 173, 160)
BUTTON_ON = (143, 122, 102)
ACCENT = (237, 194, 46)
TILE_COLOURS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
SUPER_TILE = (60, 58, 50)

# (agent name, button label, keyboard shortcut)
AGENT_BUTTONS = [
    ("expectimax", "Expectimax", pygame.K_e),
    ("minimax", "Minimax", pygame.K_m),
    ("minimax_np", "Minimax, no pruning", pygame.K_b),
    ("mcts", "MCTS", pygame.K_n),
]
ARROWS = {
    pygame.K_UP: UP,
    pygame.K_w: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_d: RIGHT,
}
DELAYS = [0.0, 0.05, 0.15, 0.4, 1.0]


@dataclass
class Stats:
    moves: int = 0
    nodes: int = 0
    seconds: float = 0.0
    best_tile: int = 0
    reached_2048: bool = False

    @property
    def nodes_per_move(self) -> float:
        return self.nodes / self.moves if self.moves else 0.0

    @property
    def ms_per_move(self) -> float:
        return 1000 * self.seconds / self.moves if self.moves else 0.0


class App:
    def __init__(self, seed: int | None = None, agent: str | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("2048 Solver")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.fonts = {size: pygame.font.Font(None, size) for size in (20, 24, 30, 40, 56, 64, 96)}

        self.seed = seed
        self.game = Game(seed=seed)
        self.best_score = 0
        self.spawned_at: dict[int, float] = {}
        self.toast: str | None = None
        self.toast_until = 0.0

        self.agent: Agent | None = None
        self.stats: dict[str, Stats] = {name: Stats() for name, _, _ in AGENT_BUTTONS}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None
        self.pending_started = 0.0
        self.delay_index = 1
        self.last_move_at = 0.0

        self.batch: list[str] = []
        self.buttons: list[tuple] = []
        self._layout_buttons()
        if agent:
            self.start_agent(agent)

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.cancel_pending()
        self.game = Game(seed=None if self.seed is None else self.seed + 1)
        if self.seed is not None:
            self.seed += 1
        self.spawned_at = {i: time.time() for i in range(16)}
        self.toast = None

    def start_agent(self, name: str) -> None:
        self.cancel_pending()
        if self.game.over:
            self.reset()
        self.agent = make_agent(name)
        self.last_move_at = 0.0

    def stop_agent(self) -> None:
        self.cancel_pending()
        self.agent = None
        self.batch = []

    def toggle_agent(self, name: str) -> None:
        if self.agent and self.agent.name == name and not self.batch:
            self.stop_agent()
        else:
            self.batch = []
            self.start_agent(name)

    def toggle_batch(self) -> None:
        if self.batch:
            self.stop_agent()
            return
        self.batch = [name for name, _, _ in AGENT_BUTTONS]
        self.reset()
        self.start_agent(self.batch[0])

    def cancel_pending(self) -> None:
        # The worker thread cannot be interrupted, but its result is ignored.
        self.pending = None

    def apply_move(self, direction: int) -> None:
        before = self.game.board
        if not self.game.play(direction):
            return
        # Remember where the new tile appeared so it can "pop" in.
        after_grid, before_grid = to_grid(self.game.board), to_grid(before)
        now = time.time()
        for i in range(16):
            r, c = divmod(i, 4)
            if before_grid[r][c] == 0 and after_grid[r][c] in (2, 4):
                self.spawned_at[i] = now
        self.best_score = max(self.best_score, self.game.score)
        if self.game.max_tile >= 2048 and max_tile(before) < 2048:
            self.show_toast("2048!")

    def show_toast(self, text: str, seconds: float = 1.5) -> None:
        self.toast, self.toast_until = text, time.time() + seconds

    # ------------------------------------------------------------- AI driving

    def update_agent(self) -> None:
        if not self.agent:
            return
        stats = self.stats[self.agent.name]

        if self.pending is None:
            if self.game.over or (self.batch and self.game.max_tile >= 2048):
                self.finish_run()
                return
            if time.time() - self.last_move_at >= DELAYS[self.delay_index]:
                agent, board = self.agent, self.game.board
                self.pending_started = time.perf_counter()
                self.pending = self.executor.submit(agent.choose, board)
            return

        if not self.pending.done():
            return
        future, self.pending = self.pending, None
        choice = future.result()
        stats.seconds += time.perf_counter() - self.pending_started
        stats.nodes += self.agent.nodes
        if choice is None:
            self.finish_run()
            return
        self.apply_move(choice)
        stats.moves += 1
        stats.best_tile = max(stats.best_tile, self.game.max_tile)
        stats.reached_2048 |= self.game.max_tile >= 2048
        self.last_move_at = time.time()

    def finish_run(self) -> None:
        if self.batch:
            self.batch.pop(0)
            if self.batch:
                self.reset()
                self.start_agent(self.batch[0])
                return
            self.show_toast("Batch finished", 3)
        self.agent = None

    # ------------------------------------------------------------------ input

    def _layout_buttons(self) -> None:
        y = GRID_Y
        for name, label, key in AGENT_BUTTONS:
            self.buttons.append((pygame.Rect(PANEL_X, y, 250, 40), name, label, key))
            y += 50
        self.buttons.append((pygame.Rect(PANEL_X, y, 250, 40), "batch", "Batch: all four to 2048", pygame.K_TAB))

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_r:
                    self.stop_agent()
                    self.reset()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.delay_index = max(0, self.delay_index - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.delay_index = min(len(DELAYS) - 1, self.delay_index + 1)
                elif event.key == pygame.K_TAB:
                    self.toggle_batch()
                elif event.key in ARROWS and not self.agent:
                    self.apply_move(ARROWS[event.key])
                else:
                    for _, name, _, key in self.buttons:
                        if event.key == key and name != "batch":
                            self.toggle_agent(name)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, name, _, _ in self.buttons:
                    if not rect.collidepoint(event.pos):
                        continue
                    if name == "batch":
                        self.toggle_batch()
                    else:
                        self.toggle_agent(name)
        return True

    # ---------------------------------------------------------------- drawing

    def text(self, value: str, size: int, colour, **anchor) -> pygame.Rect:
        surface = self.fonts[size].render(value, True, colour)
        rect = surface.get_rect(**anchor)
        self.screen.blit(surface, rect)
        return rect

    def draw(self) -> None:
        s = self.screen
        s.fill(BACKGROUND)
        self.text("2048", 96, TEXT_DARK, topleft=(GRID_X, 36))
        self.draw_score_box("SCORE", self.game.score, GRID_X + GRID_SIZE - 230)
        self.draw_score_box("BEST", self.best_score, GRID_X + GRID_SIZE - 110)
        self.draw_board()
        self.draw_panel()
        if self.toast and time.time() < self.toast_until:
            rect = self.text(self.toast, 64, TEXT_LIGHT, center=(GRID_X + GRID_SIZE // 2, GRID_Y + GRID_SIZE // 2))
            pygame.draw.rect(s, ACCENT, rect.inflate(48, 28), border_radius=10)
            self.text(self.toast, 64, TEXT_LIGHT, center=rect.center)
        pygame.display.flip()

    def draw_score_box(self, label: str, value: int, x: int) -> None:
        rect = pygame.Rect(x, 44, 110, 64)
        pygame.draw.rect(self.screen, BOARD_BG, rect, border_radius=6)
        self.text(label, 20, (238, 228, 218), center=(rect.centerx, rect.y + 16))
        self.text(f"{value:,}", 30 if value < 100_000 else 24, TEXT_LIGHT, center=(rect.centerx, rect.y + 43))

    def draw_board(self) -> None:
        s = self.screen
        pygame.draw.rect(s, BOARD_BG, (GRID_X, GRID_Y, GRID_SIZE, GRID_SIZE), border_radius=10)
        grid = to_grid(self.game.board)
        now = time.time()
        for i in range(16):
            r, c = divmod(i, 4)
            x = GRID_X + GAP + c * (CELL + GAP)
            y = GRID_Y + GAP + r * (CELL + GAP)
            pygame.draw.rect(s, EMPTY, (x, y, CELL, CELL), border_radius=6)
            value = grid[r][c]
            if not value:
                continue
            age = now - self.spawned_at.get(i, 0)
            scale = min(1.0, 0.3 + age / 0.12) if age < 0.12 else 1.0
            size = int(CELL * scale)
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (x + CELL // 2, y + CELL // 2)
            pygame.draw.rect(s, TILE_COLOURS.get(value, SUPER_TILE), rect, border_radius=6)
            digits = len(str(value))
            font = 64 if digits <= 2 else 56 if digits == 3 else 40
            colour = TEXT_DARK if value <= 4 else TEXT_LIGHT
            if scale > 0.6:
                self.text(str(value), font, colour, center=rect.center)

        if self.game.over:
            veil = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
            veil.fill((238, 228, 218, 190))
            s.blit(veil, (GRID_X, GRID_Y))
            self.text("Game over", 64, TEXT_DARK, center=(GRID_X + GRID_SIZE // 2, GRID_Y + GRID_SIZE // 2 - 20))
            self.text(
                "press R for a new game", 30, TEXT_DARK, center=(GRID_X + GRID_SIZE // 2, GRID_Y + GRID_SIZE // 2 + 30)
            )

    def draw_panel(self) -> None:
        s = self.screen
        active = self.agent.name if self.agent else None
        status = "You are playing (arrow keys / WASD)" if not self.agent else f"{self.agent.label} is playing"
        if self.batch:
            status = f"Batch run: {self.agent.label if self.agent else ''}"
        self.text(status, 30, TEXT_DARK, topleft=(PANEL_X, 60))
        self.text(
            f"Move delay {DELAYS[self.delay_index]:.2f}s  (+ / - to change)", 24, TEXT_DARK, topleft=(PANEL_X, 96)
        )

        for rect, name, label, key in self.buttons:
            on = bool(self.batch) if name == "batch" else name == active
            pygame.draw.rect(s, BUTTON_ON if on else BUTTON, rect, border_radius=6)
            hint = "Tab" if name == "batch" else pygame.key.name(key).upper()
            self.text(label, 24, TEXT_LIGHT, midleft=(rect.x + 14, rect.centery))
            self.text(hint, 24, (238, 228, 218), midright=(rect.right - 12, rect.centery))

        # Stats table
        y = GRID_Y + 5 * 50 + 30
        headers = [("Agent", 0), ("Moves", 135), ("Nodes/move", 210), ("ms/move", 330), ("Best", 415)]
        for title, dx in headers:
            self.text(title, 24, TEXT_DARK, topleft=(PANEL_X + dx, y))
        pygame.draw.line(s, BOARD_BG, (PANEL_X, y + 24), (PANEL_X + 470, y + 24), 2)
        y += 34
        for name, label, _ in AGENT_BUTTONS:
            st = self.stats[name]
            colour = BUTTON_ON if name == active else TEXT_DARK
            cells = [
                label.replace(", no pruning", " (NP)"),
                f"{st.moves:,}",
                f"{st.nodes_per_move:,.0f}",
                f"{st.ms_per_move:.1f}",
                f"{st.best_tile or '-'}" + (" *" if st.reached_2048 else ""),
            ]
            for (_, dx), cell in zip(headers, cells, strict=False):
                self.text(cell, 24, colour, topleft=(PANEL_X + dx, y))
            y += 28
        self.text("* reached 2048 this session", 20, TEXT_DARK, topleft=(PANEL_X, y + 6))
        self.text("R new game   Esc quit", 24, TEXT_DARK, bottomleft=(PANEL_X, GRID_Y + GRID_SIZE))

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.update_agent()
            self.draw()
            self.clock.tick(60)
        self.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play 2048 or watch the AI agents play it.")
    parser.add_argument("--agent", choices=[name for name, _, _ in AGENT_BUTTONS], help="start with this agent playing")
    parser.add_argument("--seed", type=int, help="seed for reproducible tile spawns")
    args = parser.parse_args(argv)
    App(seed=args.seed, agent=args.agent).run()


if __name__ == "__main__":
    main()
