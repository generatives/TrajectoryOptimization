from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush


@dataclass(slots=True)
class GridMap:
    width: int
    height: int
    cell_size: int
    cells: list[list[int]]

    @classmethod
    def empty(cls, width: int, height: int, cell_size: int) -> "GridMap":
        return cls(
            width=width,
            height=height,
            cell_size=cell_size,
            cells=[[0] * width for _ in range(height)],
        )

    @classmethod
    def from_ascii(cls, rows: list[str], cell_size: int) -> "GridMap":
        if not rows:
            raise ValueError("rows must not be empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all rows must have the same length")
        cells = [[1 if ch == "#" else 0 for ch in row] for row in rows]
        return cls(width=width, height=len(rows), cell_size=cell_size, cells=cells)

    @property
    def pixel_width(self) -> int:
        return self.width * self.cell_size

    @property
    def pixel_height(self) -> int:
        return self.height * self.cell_size
    
    def get_occupied_centers(self):
        for gx in range(self.width):
            for gy in range(self.height):
                if self.cells[gy][gx] == 1:
                    yield self.grid_to_world_center(gx, gy)

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.pixel_width and 0 <= y < self.pixel_height

    def is_occupied(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return True
        gx, gy = self.world_to_grid(x, y)
        return self.cells[gy][gx] == 1

    def set_occupied(self, x: float, y: float, occupied: bool) -> None:
        if not self.in_bounds(x, y):
            raise ValueError(f"position out of bounds: {(x, y)}")
        gx, gy = self.world_to_grid(x, y)
        self.cells[gy][gx] = 1 if occupied else 0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.cell_size), int(y // self.cell_size)

    def grid_to_world_center(self, gx: int, gy: int) -> tuple[float, float]:
        return (gx + 0.5) * self.cell_size, (gy + 0.5) * self.cell_size

    def collides_circle(self, x: float, y: float, radius: float) -> bool:
        min_gx = int((x - radius) // self.cell_size)
        max_gx = int((x + radius) // self.cell_size)
        min_gy = int((y - radius) // self.cell_size)
        max_gy = int((y + radius) // self.cell_size)

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if not self._is_occupied_cell(gx, gy):
                    continue
                cx, cy = self.grid_to_world_center(gx, gy)
                half = self.cell_size * 0.5
                closest_x = max(cx - half, min(x, cx + half))
                closest_y = max(cy - half, min(y, cy + half))
                dx = x - closest_x
                dy = y - closest_y
                if dx * dx + dy * dy <= radius * radius:
                    return True
        return False

    def astar(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        if self.is_occupied(*start) or self.is_occupied(*goal):
            return []
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)

        frontier: list[tuple[float, tuple[int, int]]] = []
        heappush(frontier, (0.0, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        cost_so_far: dict[tuple[int, int], float] = {start_cell: 0.0}

        while frontier:
            _, current = heappop(frontier)
            if current == goal_cell:
                break

            for neighbor in self._neighbors_4(current):
                new_cost = cost_so_far[current] + 1.0
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._manhattan(neighbor, goal_cell)
                    heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_cell not in came_from:
            return []

        path_cells = [goal_cell]
        node = goal_cell
        while came_from[node] is not None:
            node = came_from[node]
            path_cells.append(node)
        path_cells.reverse()
        return [self.grid_to_world_center(gx, gy) for gx, gy in path_cells]

    def _neighbors_4(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        gx, gy = node
        out: list[tuple[int, int]] = []
        for nx, ny in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
            if self._in_bounds_cell(nx, ny) and not self._is_occupied_cell(nx, ny):
                out.append((nx, ny))
        return out

    def _in_bounds_cell(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _is_occupied_cell(self, gx: int, gy: int) -> bool:
        if not self._in_bounds_cell(gx, gy):
            return True
        return self.cells[gy][gx] == 1

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
