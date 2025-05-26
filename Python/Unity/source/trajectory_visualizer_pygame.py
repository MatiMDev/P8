import pygame
import threading
from collections import deque
from typing import List, Tuple, Optional
import time
import numpy as np
from occupancy_grid import occupancy_grid

class TrajectoryVisualizerPygame:
    def __init__(self, 
                 window_size: Tuple[int, int] = (600, 600),
                 map_size_meters: float = 400.0,
                 max_points: int = 1000):
        self.window_size = window_size
        # Scale factor is computed to fit map_size_meters into window_size
        self.scale_factor = min(window_size[0], window_size[1]) / map_size_meters
        self.max_points = max_points
        # Remove maxlen to keep the entire trajectory
        self.trajectory = deque()  # Each item: (x, y, z, timestamp)
        self.running = False
        self.thread = None
        self.bg_color = (0, 0, 0)
        self.line_color = (0, 255, 255)
        self.point_color = (255, 0, 0)
        self.font_color = (0, 255, 255)
        self.font_size = 24
        self.origin: Optional[Tuple[float, float, float]] = None
        self.lock = threading.Lock()  # Add lock for thread safety

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def add_position(self, position: List[float], timestamp: float):
        if len(position) != 3:
            return
        with self.lock:  # Protect trajectory modification
            if self.origin is None:
                self.origin = tuple(position)
            rel_pos = [position[i] - self.origin[i] for i in range(3)]
            self.trajectory.append((rel_pos[0], rel_pos[1], rel_pos[2], timestamp))

    def reset(self):
        with self.lock:  # Protect trajectory modification
            self.trajectory.clear()
            self.origin = None

    def _run(self):
        pygame.init()
        screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Car Trajectory with Occupancy Grid')
        font = pygame.font.SysFont('consolas', self.font_size)
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            screen.fill(self.bg_color)

            # Draw occupancy grid
            grid_vis = occupancy_grid.get_visualization()
            if grid_vis is not None:
                # Resize grid visualization to fit the window
                grid_surface = pygame.surfarray.make_surface(grid_vis)
                grid_surface = pygame.transform.scale(grid_surface, self.window_size)
                screen.blit(grid_surface, (0, 0))

            # Draw trajectory
            with self.lock:  # Protect trajectory access
                if len(self.trajectory) > 1:
                    points = [
                        (int(self.window_size[0] / 2 + x * self.scale_factor),
                         int(self.window_size[1] / 2 - z * self.scale_factor))
                        for x, y, z, t in self.trajectory
                    ]
                    pygame.draw.lines(screen, self.line_color, False, points, 2)

                # Draw current position
                if self.trajectory:
                    x, y, z, t = self.trajectory[-1]
                    cx = int(self.window_size[0] / 2 + x * self.scale_factor)
                    cy = int(self.window_size[1] / 2 - z * self.scale_factor)
                    pygame.draw.circle(screen, self.point_color, (cx, cy), 7)
                    # Draw timestamp
                    ts_text = font.render(f't={t:.1f}s', True, self.font_color)
                    screen.blit(ts_text, (cx + 10, cy))

            # Draw title
            title_text = font.render('Car Trajectory with Occupancy Grid', True, self.font_color)
            screen.blit(title_text, (10, 10))

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

# Singleton instance
trajectory_visualizer_pygame = TrajectoryVisualizerPygame()
__all__ = ['trajectory_visualizer_pygame'] 