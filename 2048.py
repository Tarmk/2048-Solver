import pygame
import random
import time
import copy
from typing import List, Tuple, Optional, Dict, Any
import math
from concurrent.futures import ThreadPoolExecutor

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_SIZE = 4
CELL_SIZE = 120
CELL_MARGIN = 10
GRID_MARGIN = 50

# Colors
BACKGROUND = (250, 248, 239)
GRID_BACKGROUND = (205, 193, 180)
EMPTY_CELL = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)
BUTTON_BG = (187, 173, 160)
BUTTON_ACTIVE = (143, 122, 102)
BUTTON_TEXT = (255, 255, 255)

# Tile colors (value: (R, G, B))
TILE_COLORS = {
    0: (205, 193, 180),
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
    4096: (237, 191, 29),
    8192: (237, 188, 12)
}

# Heuristic weight matrix (higher scores in the top-left corner)
WEIGHT_MATRIX = [
    [65536, 32768, 16384,  8192],
    [  512,  1024,  2048,  4096],
    [  256,   128,    64,    32],
    [    2,     4,     8,    16]
]

class MCTSNode:
    def __init__(self, grid: List[List[int]], parent=None, move=None):
        self.grid = [row[:] for row in grid]
        self.parent = parent
        self.move = move  # The move that led to this state ('up', 'down', 'left', 'right')
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_moves = ['up', 'down', 'left', 'right']
        
    def add_child(self, grid: List[List[int]], move: str) -> 'MCTSNode':
        """Add a child node with the given grid state and move"""
        child = MCTSNode(grid, self, move)
        self.children.append(child)
        return child
    
    def update(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.value += result
        
    def get_ucb_score(self, exploration_constant: float) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0 or not self.parent:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class Tile:
    def __init__(self, value: int, x: int, y: int):
        self.value = value
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.current_x = float(x)  # For smooth animation
        self.current_y = float(y)  # For smooth animation
        self.merged_from = None
        self.animation_progress = 0.0
        self.scale = 1.0
        self.animation_speed = 15.0  # Speed of movement animation
        
    def update_animation(self, dt: float):
        """Update tile animation"""
        # Update scale animation
        if self.value > 0 and self.scale < 1.0:
            self.scale += dt * 10
            if self.scale > 1.0:
                self.scale = 1.0
        
        # Update position animation
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            self.current_x += dx * dt * self.animation_speed
            self.current_y += dy * dt * self.animation_speed
        
    def is_animating(self) -> bool:
        """Check if tile is still animating"""
        return (self.scale < 1.0 or 
                abs(self.current_x - self.target_x) > 0.01 or 
                abs(self.current_y - self.target_y) > 0.01)

class Game2048:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.score = 0
        self.best_score = 0
        self.game_over = False
        self.won = False
        self.can_continue = False
        self.tiles = []
        self.new_tiles = []
        self.merged_tiles = []
        
        # Setup display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2048")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Calculate grid position
        self.grid_x = (WINDOW_WIDTH - (GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * CELL_MARGIN)) // 2
        self.grid_y = 150
        
        # Initialize game
        self.add_random_tile()
        self.add_random_tile()

        # --- UI Buttons for AI modes --- #
        # Define available AI modes (label, internal type)
        button_specs = [
            ("Minimax", "minimax"),
            ("MCTS", "mcts"),
            ("Expectimax", "expectimax"),
            ("Minimax NP", "minimax_np"),  # new: naïve minimax (No Pruning)
            ("Batch", "batch")  # sequential run
        ]

        button_width = 140
        button_height = 40
        button_spacing = 20
        button_y = (self.grid_y + GRID_SIZE * (CELL_SIZE + CELL_MARGIN) + 3 * CELL_MARGIN)
        total_width = len(button_specs) * button_width + (len(button_specs) - 1) * button_spacing
        start_x = (WINDOW_WIDTH - total_width) // 2

        self.buttons = []  # List of dicts {rect,label,type}
        for i, (label, ai_type) in enumerate(button_specs):
            rect = pygame.Rect(start_x + i * (button_width + button_spacing),
                               button_y, button_width, button_height)
            self.buttons.append({"rect": rect, "label": label, "type": ai_type})
        
        # AI settings
        self.ai_enabled = False
        self.ai_type = None  # 'minimax' or 'mcts'
        self.ai_thinking = False
        self.ai_move_delay = 0.15  # Balanced speed/strength
        self.last_ai_move_time = 0
        self.minimax_depth = 4  # Increased depth for better accuracy
        self.expectimax_depth = 4  # Match depth for expectimax
        self.base_mcts_simulations = 120  # More simulations for MCTS accuracy
        # Transposition / memoisation cache shared by all tree searches
        self.transposition_cache: Dict[Any, Tuple[float, Optional[str]]] = {}
        self.show_ai_info = True  # Show AI evaluation info
        self.current_evaluation = None
        self.current_depth = 0
        self.nodes_explored = 0

        # Batch / sequential run helpers
        self.batch_sequence = ['minimax', 'mcts', 'expectimax', 'minimax_np']
        self.batch_mode = False
        self.batch_index = 0

        # Metrics per algorithm
        self.metrics = {
            'minimax':     {'nodes': 0, 'time': 0.0, 'moves': 0, 'won': False},
            'mcts':        {'nodes': 0, 'time': 0.0, 'moves': 0, 'won': False},
            'expectimax':  {'nodes': 0, 'time': 0.0, 'moves': 0, 'won': False},
            'minimax_np':  {'nodes': 0, 'time': 0.0, 'moves': 0, 'won': False},  # naive minimax
        }
        
    def get_empty_cells(self, grid: Optional[List[List[int]]] = None) -> List[Tuple[int, int]]:
        """Get list of empty cell positions for the given grid (defaults to self.grid)"""
        if grid is None:
            grid = self.grid
        empty = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] == 0:
                    empty.append((x, y))
        return empty
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to the grid"""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            x, y = random.choice(empty_cells)
            value = 2 if random.random() < 0.9 else 4
            self.grid[y][x] = value
            
            # Create tile object for animation
            tile = Tile(value, x, y)
            tile.scale = 0.0  # Start small for spawn animation
            self.tiles.append(tile)

    # ---------------------------- Tile helpers ---------------------------- #
    def rebuild_tiles_from_grid(self, grid: Optional[List[List[int]]] = None):
        """Recreate the Tile list to match the current grid state (no animation)"""
        if grid is None:
            grid = self.grid
        self.tiles = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                value = grid[y][x]
                if value != 0:
                    self.tiles.append(Tile(value, x, y))

    # ------------------------------------------------------------------
    # Game reset that preserves accumulated metrics
    # ------------------------------------------------------------------
    def reset_board(self):
        """Reset the board and scores but keep algorithm metrics intact."""
        # Logical grid & score
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.score = 0
        self.game_over = False
        self.won = False

        # Visual tiles list
        self.tiles = []
        self.add_random_tile()
        self.add_random_tile()

        # AI state
        self.ai_enabled = False
        self.ai_type = None
        self.current_evaluation = None
        self.current_depth = 0
        self.nodes_explored = 0
        self.last_ai_move_time = time.time()

    def check_win_condition(self):
        """Check if a 2048 tile exists and end the game if so"""
        if not self.won:
            for row in self.grid:
                if 2048 in row:
                    self.won = True
                    self.game_over = True
                    # Mark win in metrics immediately
                    if self.ai_enabled and self.ai_type in self.metrics:
                        self.metrics[self.ai_type]['won'] = True
                    break
    
    def evaluate_position(self, grid: List[List[int]]) -> float:
        """Evaluate a grid position for AI"""
        score = 0.0
        empty_cells = 0
        max_tile = 0
        merge_potential = 0
        weight_score = 0

        # Smoothness & monotonicity helpers
        def smooth_difference(a: int, b: int) -> int:
            return abs(a - b)

        def row_monotone(row: List[int]) -> int:
            score = 0
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    score += 1
                else:
                    score -= 1
            return score

        smooth_penalty = 0
        monotone_score = 0
        
        # First, find the maximum tile and count empty cells
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] == 0:
                    empty_cells += 1
                else:
                    max_tile = max(max_tile, grid[y][x])
                    # Weighted positional score (corner strategy)
                    weight_score += grid[y][x] * WEIGHT_MATRIX[y][x]

        # Smoothness (difference between neighbouring tiles)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] == 0:
                    continue
                v = grid[y][x]
                for dx, dy in [(1,0),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] != 0:
                        smooth_penalty += smooth_difference(v, grid[ny][nx])

        # Monotonicity rows & columns
        for row in grid:
            monotone_score += row_monotone(row)
        for col in zip(*grid):
            monotone_score += row_monotone(list(col))
        
        # Evaluate merge potential and positioning
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] != 0:
                    current = grid[y][x]
                    
                    # Check for mergeable tiles (same values)
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            if grid[ny][nx] == current:
                                # Higher value merges are more important
                                merge_potential += current * 2
                    
                    # Encourage keeping high values in corners
                    if (x == 0 or x == GRID_SIZE-1) and (y == 0 or y == GRID_SIZE-1):
                        if current >= max_tile / 2:  # High value tiles
                            score += current * 4
                    
                    # Encourage keeping similar values adjacent
                    for dx, dy in [(0,1), (1,0)]:  # Only check right and down
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] != 0:
                            # Reward having powers of 2 next to each other in sequence
                            current_power = math.log2(current)
                            neighbor_power = math.log2(grid[ny][nx])
                            if abs(current_power - neighbor_power) <= 1:
                                score += min(current, grid[ny][nx])
        
        # Weight the components
        final_score = 0.0
        final_score += weight_score * 1.0  # Dominant positional term
        final_score += empty_cells * 500.0  # Encourage open spaces
        final_score += merge_potential * 50.0  # Immediate merges
        final_score += math.pow(max_tile, 2.1)   # Heavily reward big tiles
        final_score += monotone_score * 100.0
        final_score -= smooth_penalty * 3.0
        final_score += score  # Adjacency / extra helpers
        
        return final_score
    
    def get_valid_moves(self, grid: List[List[int]]) -> List[str]:
        """Get list of valid moves"""
        valid_moves = []
        for move in ['up', 'down', 'left', 'right']:
            test_grid = [row[:] for row in grid]
            if self.make_move(test_grid, move, test=True):
                valid_moves.append(move)
        return valid_moves

    # ------------------------------------------------------------------
    # Fast move-existence check (empties or adjacent equals)
    # ------------------------------------------------------------------
    def has_any_moves(self, grid: Optional[List[List[int]]] = None) -> bool:
        """Return True if at least one legal move exists (faster than simulating)."""
        if grid is None:
            grid = self.grid
        # Empty cell => move available
        for row in grid:
            if 0 in row:
                return True
        # Adjacent equal horizontally/vertically
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                v = grid[y][x]
                if x + 1 < GRID_SIZE and grid[y][x + 1] == v:
                    return True
                if y + 1 < GRID_SIZE and grid[y + 1][x] == v:
                    return True
        return False
    
    def make_move(self, grid: List[List[int]], direction: str, test: bool = False) -> bool:
        """Make a move in the specified direction. Returns True if grid changed."""
        if direction == 'up':
            return self.move_up(grid, test)
        elif direction == 'down':
            return self.move_down(grid, test)
        elif direction == 'left':
            return self.move_left(grid, test)
        elif direction == 'right':
            return self.move_right(grid, test)
        return False
    
    def merge_line(self, line: List[int]) -> Tuple[List[int], int]:
        """Merge a line of tiles, return new line and score gained"""
        # Remove zeros
        line = [x for x in line if x != 0]
        score = 0
        
        # Merge adjacent equal tiles
        i = 0
        while i < len(line) - 1:
            if line[i] == line[i + 1]:
                line[i] *= 2
                score += line[i]
                line.pop(i + 1)
            i += 1
            
        # Pad with zeros
        while len(line) < GRID_SIZE:
            line.append(0)
            
        return line, score
    
    def move_up(self, grid: List[List[int]], test: bool = False) -> bool:
        """Move all tiles up"""
        moved = False
        for x in range(GRID_SIZE):
            # Get column
            column = [grid[y][x] for y in range(GRID_SIZE)]
            # Merge
            merged, score = self.merge_line(column)
            # Check if anything moved
            if column != merged:
                moved = True
                # Update grid
                for y in range(GRID_SIZE):
                    grid[y][x] = merged[y]
                if not test:
                    self.score += score

        if moved and not test and grid is self.grid:
            # Rebuild visible tiles to reflect new positions
            self.rebuild_tiles_from_grid()
            self.check_win_condition()
        return moved
    
    def move_down(self, grid: List[List[int]], test: bool = False) -> bool:
        """Move all tiles down"""
        moved = False
        for x in range(GRID_SIZE):
            # Get column
            column = [grid[y][x] for y in range(GRID_SIZE)]
            # Reverse for down movement
            column.reverse()
            # Merge
            merged, score = self.merge_line(column)
            # Reverse back
            merged.reverse()
            # Check if anything moved
            if column != merged:
                moved = True
                # Update grid
                for y in range(GRID_SIZE):
                    grid[y][x] = merged[y]
                if not test:
                    self.score += score

        if moved and not test and grid is self.grid:
            self.rebuild_tiles_from_grid()
            self.check_win_condition()
        return moved
    
    def move_left(self, grid: List[List[int]], test: bool = False) -> bool:
        """Move all tiles left"""
        moved = False
        for y in range(GRID_SIZE):
            # Get row
            row = grid[y][:]
            # Merge
            merged, score = self.merge_line(row)
            # Check if anything moved
            if row != merged:
                moved = True
                # Update grid
                grid[y] = merged
                if not test:
                    self.score += score

        if moved and not test and grid is self.grid:
            self.rebuild_tiles_from_grid()
            self.check_win_condition()
        return moved
    
    def move_right(self, grid: List[List[int]], test: bool = False) -> bool:
        """Move all tiles right"""
        moved = False
        for y in range(GRID_SIZE):
            # Get row
            row = grid[y][:]
            # Reverse for right movement
            row.reverse()
            # Merge
            merged, score = self.merge_line(row)
            # Reverse back
            merged.reverse()
            # Check if anything moved
            if row != merged:
                moved = True
                # Update grid
                grid[y] = merged
                if not test:
                    self.score += score

        if moved and not test and grid is self.grid:
            self.rebuild_tiles_from_grid()
            self.check_win_condition()
        return moved
    
    def minimax(self, grid: List[List[int]], depth: int, is_maximizing: bool,
                alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[float, Optional[str]]:
        """Minimax algorithm with alpha-beta pruning and transposition caching"""
        self.nodes_explored += 1
        # Use transposition cache
        grid_key = tuple(tuple(row) for row in grid)
        cache_key = (grid_key, depth, is_maximizing)
        cached = self.transposition_cache.get(cache_key)
        if cached:
            return cached
        
        self.current_depth = self.minimax_depth - depth
        
        if depth == 0:
            return self.evaluate_position(grid), None
        
        if is_maximizing:
            max_eval = float('-inf')
            best_move = None
            # Evaluate each move once using a fresh grid copy
            for move in ['up', 'down', 'left', 'right']:
                new_grid = [row[:] for row in grid]
                if not self.make_move(new_grid, move, test=True):
                    continue  # Skip illegal move

                # Calculate merge score for ordering (optional heuristic)
                merge_score = 0
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        if new_grid[y][x] > grid[y][x] and grid[y][x] != 0:
                            merge_score += new_grid[y][x]

                eval_score, _ = self.minimax(new_grid, depth - 1, False, alpha, beta)

                # Slightly bias toward moves that merge big tiles when scores tie
                eval_score += merge_score * 0.01

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            result = (max_eval, best_move)
            self.transposition_cache[cache_key] = result
            return result
        else:
            # Simulate opponent (random tile placement)
            min_eval = float('inf')
            empty_cells = self.get_empty_cells(grid)
            
            if not empty_cells:
                return self.evaluate_position(grid), None
            
            # Consider all empty cells for accurate evaluation
            for x, y in empty_cells:
                # 90% chance of 2, 10% chance of 4
                new_grid_2 = [row[:] for row in grid]
                new_grid_2[y][x] = 2
                eval_2, _ = self.minimax(new_grid_2, depth - 1, True, alpha, beta)
                
                new_grid_4 = [row[:] for row in grid]
                new_grid_4[y][x] = 4
                eval_4, _ = self.minimax(new_grid_4, depth - 1, True, alpha, beta)
                
                # Weight the evaluations by their probabilities
                eval_score = (eval_2 * 0.9 + eval_4 * 0.1)
                min_eval = min(min_eval, eval_score)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            result = (min_eval, None)
            self.transposition_cache[cache_key] = result
            return result

    # ------------------------------------------------------------------
    # Naïve minimax (no alpha-beta pruning)
    # ------------------------------------------------------------------
    def minimax_naive(self, grid: List[List[int]], depth: int, is_maximizing: bool) -> Tuple[float, Optional[str]]:
        """Plain minimax search without alpha-beta pruning."""
        self.nodes_explored += 1

        grid_key = tuple(tuple(row) for row in grid)
        cache_key = (grid_key, depth, is_maximizing, 'naive')
        cached = self.transposition_cache.get(cache_key)
        if cached:
            return cached

        # Track depth for UI display (same convention as pruned minimax)
        self.current_depth = self.minimax_depth - depth

        # Terminal node
        if depth == 0:
            result = (self.evaluate_position(grid), None)
            self.transposition_cache[cache_key] = result
            return result

        if is_maximizing:
            best_val = float('-inf')
            best_move = None
            for move in ['up', 'down', 'left', 'right']:
                new_grid = [row[:] for row in grid]
                if not self.make_move(new_grid, move, test=True):
                    continue
                val, _ = self.minimax_naive(new_grid, depth - 1, False)
                if val > best_val:
                    best_val = val
                    best_move = move
            result = (best_val, best_move)
        else:
            # Chance/minimising layer: evaluate all possible random tile insertions
            worst_val = float('inf')
            empty = self.get_empty_cells(grid)
            if not empty:
                result = (self.evaluate_position(grid), None)
            else:
                # Use all empty cells for more accurate expectation
                for (x, y) in empty:
                    # 90% 2, 10% 4
                    grid2 = [row[:] for row in grid]
                    grid2[y][x] = 2
                    val2, _ = self.minimax_naive(grid2, depth - 1, True)

                    grid4 = [row[:] for row in grid]
                    grid4[y][x] = 4
                    val4, _ = self.minimax_naive(grid4, depth - 1, True)

                    expected = val2 * 0.9 + val4 * 0.1
                    worst_val = min(worst_val, expected)
                result = (worst_val, None)

        self.transposition_cache[cache_key] = result
        return result

    def expectimax(self, grid: List[List[int]], depth: int, is_max_node: bool,
                   alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[float, Optional[str]]:
        """Expectimax search (max node vs chance node). Alpha/beta used only for max node pruning."""
        self.nodes_explored += 1
        grid_key = tuple(tuple(row) for row in grid)
        cache_key = (grid_key, depth, is_max_node, 'exp')
        cached = self.transposition_cache.get(cache_key)
        if cached:
            return cached

        if depth == 0:
            result = (self.evaluate_position(grid), None)
            self.transposition_cache[cache_key] = result
            return result

        if is_max_node:
            best_value = float('-inf')
            best_move = None
            moves = ['up', 'down', 'left', 'right']

            def evaluate_move(move_dir: str):
                test_grid = [row[:] for row in grid]
                if not self.make_move(test_grid, move_dir, test=True):
                    return (float('-inf'), move_dir)
                val, _ = self.expectimax(test_grid, depth - 1, False, alpha, beta)
                return (val, move_dir)

            # Parallel evaluation only at root (depth == self.minimax_depth)
            if depth == self.minimax_depth:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(evaluate_move, moves))
            else:
                results = [evaluate_move(m) for m in moves]

            for val, mv in results:
                if val > best_value:
                    best_value = val
                    best_move = mv
                alpha = max(alpha, val)
            if best_move is None:
                best_value = self.evaluate_position(grid)
            result = (best_value, best_move)
        else:
            # Chance node: average over possible random tiles
            empty = self.get_empty_cells(grid)
            if not empty:
                result = (self.evaluate_position(grid), None)
            else:
                total_expectation = 0.0
                probability_sum = 0.0
                # For speed, sample up to 4 empty cells
                sample = empty if len(empty) <= 3 else random.sample(empty, 3)
                for (x, y) in sample:
                    for value, prob in [(2, 0.9), (4, 0.1)]:
                        new_grid = [row[:] for row in grid]
                        new_grid[y][x] = value
                        val, _ = self.expectimax(new_grid, depth - 1, True, alpha, beta)
                        total_expectation += prob * val
                        probability_sum += prob
                expected_value = total_expectation / probability_sum if probability_sum else 0.0
                result = (expected_value, None)

        self.transposition_cache[cache_key] = result
        return result
    
    def mcts_search(self, grid: List[List[int]], num_simulations: int) -> str:
        """Monte Carlo Tree Search for finding the best move"""
        # Adapt simulation count to board emptiness (more empties => larger branching)
        empty_cells_count = len(self.get_empty_cells(grid))
        num_simulations = int(self.base_mcts_simulations + empty_cells_count * 4)
        root = MCTSNode(grid)
        exploration_constant = 1.414  # UCT exploration parameter
        
        for _ in range(num_simulations):
            self.nodes_explored += 1  # Track simulations as nodes
            # Selection (tree policy)
            node = root
            while not node.untried_moves and node.children:
                node = max(node.children,
                           key=lambda n: n.get_ucb_score(exploration_constant))
             
            # Expansion
            if node.untried_moves:
                # Prefer a valid move among the still-untried ones
                random.shuffle(node.untried_moves)
                chosen_move = None
                for move in list(node.untried_moves):
                    test_grid = [row[:] for row in node.grid]
                    if self.make_move(test_grid, move, test=True):
                        chosen_move = move
                        break
                    else:
                        # Remove invalid moves permanently so we don't revisit them
                        node.untried_moves.remove(move)

                if chosen_move is not None:
                    node.untried_moves.remove(chosen_move)
                    new_grid = test_grid  # From earlier validity check

                    # Add random tile after player's move (environment step)
                    empty_cells = self.get_empty_cells(new_grid)
                    if empty_cells:
                        x, y = random.choice(empty_cells)
                        value = 2 if random.random() < 0.9 else 4
                        new_grid[y][x] = value

                    node = node.add_child(new_grid, chosen_move)
             
            # Simulation
            sim_grid = [row[:] for row in node.grid]
            sim_score = self.simulate_heuristic_playout(sim_grid)
             
            # Backpropagation
            while node:
                node.update(sim_score)
                node = node.parent
         
        # Choose best move
        if root.children:
            # Use highest average value; visits as tiebreaker
            best_child = max(
                root.children,
                key=lambda n: (n.value / n.visits if n.visits else float('-inf'), n.visits))
            return best_child.move

        # Fallback: choose a random VALID move
        valid = self.get_valid_moves(grid)
        return random.choice(valid) if valid else 'up'

    def simulate_random_playout(self, grid: List[List[int]]) -> float:
        """Simulate a random playout from given grid state"""
        moves_made = 0
        max_moves = 100  # Prevent infinite loops
        
        while moves_made < max_moves:
            valid_moves = self.get_valid_moves(grid)
            if not valid_moves:
                break
            
            # Make random move
            move = random.choice(valid_moves)
            self.make_move(grid, move, test=True)
            
            # Add random tile
            empty_cells = self.get_empty_cells(grid)
            if not empty_cells:
                break
            
            x, y = random.choice(empty_cells)
            value = 2 if random.random() < 0.9 else 4
            grid[y][x] = value
            
            moves_made += 1
        
        return self.evaluate_position(grid)

    def simulate_heuristic_playout(self, grid: List[List[int]]) -> float:
        """Simulate a playout using light heuristics for better guidance"""
        moves_made = 0
        max_moves = 60  # longer playout for accuracy
        while moves_made < max_moves:
            valid_moves = self.get_valid_moves(grid)
            if not valid_moves:
                break
            # Evaluate up to two sampled moves for balanced accuracy
            sample_size = min(2, len(valid_moves))
            sampled_moves = random.sample(valid_moves, sample_size)
            chosen_grid = None
            for move in sampled_moves:
                test_grid = [row[:] for row in grid]
                if self.make_move(test_grid, move, test=True):
                    chosen_grid = test_grid
                    break
            if chosen_grid is None:
                break
            grid[:] = [row[:] for row in chosen_grid]

            # Add random tile as in game
            empty = self.get_empty_cells(grid)
            if not empty:
                break
            x, y = random.choice(empty)
            grid[y][x] = 2 if random.random() < 0.9 else 4

            moves_made += 1
        return self.evaluate_position(grid)
    
    def get_ai_move(self) -> Optional[str]:
        """Get the best move using the selected AI algorithm"""
        if not self.ai_enabled or self.game_over:
            return None
        
        self.nodes_explored = 0
        self.current_evaluation = None

        # If there are no valid moves, deactivate AI immediately
        if not self.get_valid_moves(self.grid):
            self.ai_enabled = False
            self.game_over = True
            return None
        
        if self.ai_type == 'minimax':
            eval_score, best_move = self.minimax(self.grid, self.minimax_depth, True)
            self.current_evaluation = eval_score
            return best_move if best_move in self.get_valid_moves(self.grid) else None
        elif self.ai_type == 'mcts':
            # Depth label not meaningful for MCTS; set current_depth to 0
            self.current_depth = 0
            candidate = self.mcts_search(self.grid, self.base_mcts_simulations)
            return candidate if candidate in self.get_valid_moves(self.grid) else None
        elif self.ai_type == 'expectimax':
            self.current_depth = 0
            val, best_move = self.expectimax(self.grid, self.expectimax_depth, True)
            self.current_evaluation = val
            return best_move if best_move in self.get_valid_moves(self.grid) else None
        elif self.ai_type == 'minimax_np':
            eval_score, best_move = self.minimax_naive(self.grid, self.minimax_depth, True)
            self.current_evaluation = eval_score
            return best_move if best_move in self.get_valid_moves(self.grid) else None
        
        return None
    
    def update(self, dt: float):
        """Update game state"""
        # Update tile animations
        for tile in self.tiles:
            tile.update_animation(dt)
        
        # Remove merged tiles that finished animating
        self.tiles = [tile for tile in self.tiles if not tile.merged_from or tile.is_animating()]
        
        # Handle AI moves
        if self.ai_enabled and not self.game_over:
            current_time = time.time()
            if current_time - self.last_ai_move_time >= self.ai_move_delay:
                start = time.time()
                move = self.get_ai_move()
                search_time = time.time() - start
                if move:
                    moved = self.make_move(self.grid, move)
                    if moved:
                        self.add_random_tile()
                        # Record metrics only when a real move happened
                        if self.ai_type in self.metrics:
                            m = self.metrics[self.ai_type]
                            m['nodes'] += self.nodes_explored
                            m['time'] += search_time
                            m['moves'] += 1
                            if self.won:
                                m['won'] = True
                        # Update last move time regardless to avoid rapid retries
                        self.last_ai_move_time = current_time
                else:
                    # move is None -> disable AI; loss detection will decide game_over
                    self.ai_enabled = False

        # ---------------- Loss detection -----------------
        # If there are no moves left, mark game over so AI stops accumulating stats
        if not self.game_over and not self.has_any_moves():
            self.game_over = True
        # Ensure win flag propagated even if metrics not updated this frame
        if self.won and self.ai_type in self.metrics:
            self.metrics[self.ai_type]['won'] = True

        # ---------------- Batch progression -----------------
        if self.batch_mode and self.game_over:
            # advance to next algorithm
            self.batch_index += 1
            if self.batch_index < len(self.batch_sequence):
                self.ai_type = self.batch_sequence[self.batch_index]
                self.reset_board()
                self.ai_enabled = True
                self.last_ai_move_time = time.time()
                # ensure won flag resets for new run
                self.won = False
                self.game_over = False
            else:
                # finished batch
                self.batch_mode = False
                self.ai_enabled = False
    
    def draw(self):
        """Draw the game state"""
        # Draw background
        self.screen.fill(BACKGROUND)
        
        # Draw grid background with subtle rounded corners
        pygame.draw.rect(
            self.screen,
            GRID_BACKGROUND,
            (
                self.grid_x - CELL_MARGIN,
                self.grid_y - CELL_MARGIN,
                GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * CELL_MARGIN,
                GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * CELL_MARGIN,
            ),
            border_radius=8,
        )

        # Draw empty cells (so they are visible when no tile occupies them)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell_x = self.grid_x + x * (CELL_SIZE + CELL_MARGIN)
                cell_y = self.grid_y + y * (CELL_SIZE + CELL_MARGIN)
                pygame.draw.rect(self.screen, EMPTY_CELL,
                                 (cell_x, cell_y, CELL_SIZE, CELL_SIZE))
        
        # -------------------- Score / Best boxes -------------------- #
        box_width, box_height = 160, 60
        score_rect = pygame.Rect(20, 20, box_width, box_height)
        best_rect  = pygame.Rect(20, 20 + box_height + 10, box_width, box_height)

        # Draw filled rounded rectangles
        for rect in (score_rect, best_rect):
            pygame.draw.rect(self.screen, BUTTON_BG, rect, border_radius=8)

        # Score box labels and value
        score_label = self.font_small.render("SCORE", True, BUTTON_TEXT)
        score_value = self.font_large.render(f"{self.score}", True, BUTTON_TEXT)
        self.screen.blit(score_label, score_label.get_rect(center=(score_rect.centerx, score_rect.centery - 12)))
        self.screen.blit(score_value, score_value.get_rect(center=(score_rect.centerx, score_rect.centery + 10)))

        # Best box labels and value
        best_label  = self.font_small.render("BEST", True, BUTTON_TEXT)
        best_value  = self.font_medium.render(f"{self.best_score}", True, BUTTON_TEXT)
        self.screen.blit(best_label, best_label.get_rect(center=(best_rect.centerx, best_rect.centery - 12)))
        self.screen.blit(best_value, best_value.get_rect(center=(best_rect.centerx, best_rect.centery + 10)))
        
        # Draw AI info (current)
        if self.ai_enabled and self.show_ai_info and self.ai_type:
            ai_text = self.font_medium.render(
                f"AI: {self.ai_type.upper()}", True, TEXT_DARK)
            self.screen.blit(ai_text, (WINDOW_WIDTH - 150, 20))
            
            if self.current_evaluation is not None:
                eval_text = self.font_small.render(
                    f"Eval: {self.current_evaluation:.1f}", True, TEXT_DARK)
                self.screen.blit(eval_text, (WINDOW_WIDTH - 150, 60))
            
            depth_text = self.font_small.render(
                f"Depth: {self.current_depth}/{self.minimax_depth}", True, TEXT_DARK)
            self.screen.blit(depth_text, (WINDOW_WIDTH - 150, 90))
            
            nodes_text = self.font_small.render(
                f"Nodes: {self.nodes_explored}", True, TEXT_DARK)
            self.screen.blit(nodes_text, (WINDOW_WIDTH - 150, 120))
        
        # Draw AI mode buttons
        for btn in self.buttons:
            is_active = self.ai_enabled and self.ai_type == btn["type"]
            color = BUTTON_ACTIVE if is_active else BUTTON_BG
            pygame.draw.rect(self.screen, color, btn["rect"], border_radius=6)
            label_surf = self.font_small.render(btn["label"], True, BUTTON_TEXT)
            label_rect = label_surf.get_rect(center=btn["rect"].center)
            self.screen.blit(label_surf, label_rect)

        # Draw right-side metrics table
        panel_x = self.grid_x + GRID_SIZE * (CELL_SIZE + CELL_MARGIN) + 40
        panel_y = self.grid_y
        line_height = 22

        title = self.font_medium.render("Algorithm Stats", True, TEXT_DARK)
        self.screen.blit(title, (panel_x, panel_y))

        header = self.font_small.render("Algo        Nodes   Time   Win", True, TEXT_DARK)
        self.screen.blit(header, (panel_x, panel_y + 30))
        offset = panel_y + 50

        # Prepare checkbox registry
        self.metric_checkboxes = {}

        for algo, data in self.metrics.items():
            total_nodes = data['nodes']
            total_time = data['time']
            status_symbol = '✓' if data.get('won') else '✗'
            # Draw text up to Win column
            text_part = f"{algo:<10}{total_nodes:>7}  {total_time:>5.2f}s  "
            txt_surf = self.font_small.render(text_part, True, TEXT_DARK)
            self.screen.blit(txt_surf, (panel_x, offset))

            # Draw checkbox for Win status
            cb_size = 14
            cb_x = panel_x + txt_surf.get_width() + 5
            cb_y = offset + (self.font_small.get_height() - cb_size) // 2
            cb_rect = pygame.Rect(cb_x, cb_y, cb_size, cb_size)
            pygame.draw.rect(self.screen, TEXT_DARK, cb_rect, 2)
            if data.get('won'):
                # Draw tick mark
                pygame.draw.line(self.screen, TEXT_DARK, (cb_x+3, cb_y+cb_size//2), (cb_x+cb_size//3, cb_y+cb_size-3), 2)
                pygame.draw.line(self.screen, TEXT_DARK, (cb_x+cb_size//3, cb_y+cb_size-3), (cb_x+cb_size-3, cb_y+3), 2)

            # Save rect for input handling
            self.metric_checkboxes[algo] = cb_rect
            offset += line_height
        
        # Draw tiles
        for tile in self.tiles:
            if tile.value == 0:
                continue
                
            # Calculate display position
            x = self.grid_x + tile.current_x * (CELL_SIZE + CELL_MARGIN)
            y = self.grid_y + tile.current_y * (CELL_SIZE + CELL_MARGIN)
            
            # Apply scale animation
            scaled_size = int(CELL_SIZE * tile.scale)
            x_offset = (CELL_SIZE - scaled_size) // 2
            y_offset = (CELL_SIZE - scaled_size) // 2
            
            # Draw tile shadow for depth effect
            shadow_surf = pygame.Surface((scaled_size, scaled_size), pygame.SRCALPHA)
            shadow_surf.fill((0, 0, 0, 50))  # semi-transparent shadow
            self.screen.blit(shadow_surf, (x + x_offset + 4, y + y_offset + 4))

            # Draw tile background with rounded corners
            color = TILE_COLORS.get(tile.value, TILE_COLORS[0])
            rect = pygame.Rect(x + x_offset, y + y_offset, scaled_size, scaled_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

            # Outline for better visibility on light tiles (match rounded corners)
            outline_color = (150, 140, 130)
            pygame.draw.rect(self.screen, outline_color, rect, 2, border_radius=6)
            
            # Draw tile value
            if tile.value > 0:
                text_color = TEXT_DARK if tile.value < 8 else TEXT_LIGHT
                text = self.font_large.render(str(tile.value), True, text_color)
                text_rect = text.get_rect(center=(x + CELL_SIZE/2,
                                                y + CELL_SIZE/2))
                self.screen.blit(text, text_rect)
        
        # Draw game over / win overlay
        if self.game_over or self.won:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.fill((255, 255, 255))
            overlay.set_alpha(180)
            self.screen.blit(overlay, (0, 0))
            if self.won:
                win_text = self.font_large.render("You Win!", True, TEXT_DARK)
                text_rect = win_text.get_rect(center=(WINDOW_WIDTH/2,
                                                     WINDOW_HEIGHT/2))
                self.screen.blit(win_text, text_rect)
            else:
                game_over_text = self.font_large.render("Game Over!", True, TEXT_DARK)
                text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH/2,
                                                           WINDOW_HEIGHT/2))
                self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    def handle_input(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if not self.ai_enabled:
                    if event.key == pygame.K_UP:
                        if self.make_move(self.grid, 'up'):
                            self.add_random_tile()
                    elif event.key == pygame.K_DOWN:
                        if self.make_move(self.grid, 'down'):
                            self.add_random_tile()
                    elif event.key == pygame.K_LEFT:
                        if self.make_move(self.grid, 'left'):
                            self.add_random_tile()
                    elif event.key == pygame.K_RIGHT:
                        if self.make_move(self.grid, 'right'):
                            self.add_random_tile()
                
                # AI controls
                if event.key == pygame.K_m:
                    self.ai_enabled = not self.ai_enabled
                    self.ai_type = 'minimax'
                    self.last_ai_move_time = time.time()
                elif event.key == pygame.K_n:
                    self.ai_enabled = not self.ai_enabled
                    self.ai_type = 'mcts'
                    self.last_ai_move_time = time.time()
                elif event.key == pygame.K_e:
                    self.ai_enabled = not self.ai_enabled
                    self.ai_type = 'expectimax'
                    self.last_ai_move_time = time.time()
                elif event.key == pygame.K_b:  # toggle naïve minimax
                    self.ai_enabled = not self.ai_enabled
                    self.ai_type = 'minimax_np'
                    self.last_ai_move_time = time.time()
                elif event.key == pygame.K_r:
                    self.reset_board()
                elif event.key == pygame.K_a:  # start/stop batch run
                    if self.batch_mode:
                        self.batch_mode = False
                        self.ai_enabled = False
                        self.ai_type = None
                    else:
                        self.batch_mode = True
                        self.batch_index = 0
                        self.ai_type = self.batch_sequence[0]
                        self.reset_board()
                        self.ai_enabled = True
                        self.last_ai_move_time = time.time()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                for btn in self.buttons:
                    if btn["rect"].collidepoint(mouse_pos):
                        # Toggle AI for the selected type
                        if btn["type"] == 'batch':
                            # Handle batch separately
                            if self.batch_mode:
                                self.batch_mode = False
                                self.ai_enabled = False
                                self.ai_type = None
                            else:
                                self.batch_mode = True
                                self.batch_index = 0
                                self.ai_type = self.batch_sequence[0]
                                self.reset_board()
                                self.ai_enabled = True
                                self.last_ai_move_time = time.time()
                            break

                        if self.ai_enabled and self.ai_type == btn["type"]:
                            # Turn off AI if clicking active button
                            self.ai_enabled = False
                            self.ai_type = None
                        else:
                            self.ai_enabled = True
                            self.ai_type = btn["type"]
                            self.last_ai_move_time = time.time()
                        break

                # Check metric checkboxes click
                if hasattr(self, 'metric_checkboxes'):
                    for algo, rect in self.metric_checkboxes.items():
                        if rect.collidepoint(mouse_pos):
                            # Toggle win flag manually
                            self.metrics[algo]['won'] = not self.metrics[algo]['won']
                            break
        
        return True
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0  # Convert to seconds
            
            running = self.handle_input()
            self.update(dt)
            self.draw()

if __name__ == "__main__":
    game = Game2048()
    game.run() 