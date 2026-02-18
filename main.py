from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import pygame
import numpy as np

from map import GridMap
from trajoptim import TrajectoryOptimizer

CELL_SIZE = 32
GRID_W = 28
GRID_H = 20
WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE
FPS = 60

BG_COLOR = (15, 22, 35)
GRID_COLOR = (37, 49, 70)
OBSTACLE_COLOR = (70, 82, 108)
ROBOT_COLOR = (60, 205, 125)
HEADING_COLOR = (9, 28, 15)
PATH_COLOR = (255, 196, 94)
TEXT_COLOR = (219, 227, 238)

class ControlMode(IntEnum):
    TeleOp = 0
    AutoNav = 1
    TrajOptim = 2

    def next(self):
        max_value = max(e.value for e in ControlMode) + 1
        return ControlMode((self.value + 1) % max_value)


def make_demo_map() -> GridMap:
    grid = GridMap.empty(GRID_W, GRID_H, CELL_SIZE)

    def fill_cell(gx: int, gy: int) -> None:
        px, py = grid.grid_to_world_center(gx, gy)
        grid.set_occupied(px, py, True)

    for x in range(GRID_W):
        fill_cell(x, 0)
        fill_cell(x, GRID_H - 1)
    for y in range(GRID_H):
        fill_cell(0, y)
        fill_cell(GRID_W - 1, y)

    for x in range(4, 13):
        fill_cell(x, 5)
    for y in range(6, 15):
        fill_cell(10, y)
    for x in range(14, 24):
        fill_cell(x, 12)
    for y in range(3, 11):
        fill_cell(20, y)
    for x in range(5, 9):
        fill_cell(x, 15)

    return grid


@dataclass(slots=True)
class DifferentialDriveRobot:
    x: float
    y: float
    theta: float
    radius: float = 10.0
    max_v: float = 150.0
    max_omega: float = 3.0
    v: float = 0.0
    omega: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)

    def step(self, dt: float, world: GridMap) -> None:
        v = max(-self.max_v, min(self.max_v, self.v))
        omega = max(-self.max_omega, min(self.max_omega, self.omega))
        nx = self.x + v * math.cos(self.theta) * dt
        ny = self.y + v * math.sin(self.theta) * dt
        ntheta = self._wrap_angle(self.theta + omega * dt)

        if not world.collides_circle(nx, ny, self.radius):
            self.x, self.y = nx, ny

        self.theta = ntheta
        self.trail.append((self.x, self.y))
        if len(self.trail) > 350:
            self.trail = self.trail[-350:]

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


def waypoints_to_commands(
    waypoints: list[tuple[float, float]],
    start_x: float,
    start_y: float,
    start_theta: float,
    dt: float,
    max_v: float,
    max_omega: float,
    waypoint_tolerance: float = 10.0,
    max_steps: int = 2000,
) -> list[tuple[float, float]]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not waypoints:
        return []

    x = start_x
    y = start_y
    theta = start_theta
    waypoint_index = 0
    commands: list[tuple[float, float]] = []

    for _ in range(max_steps):
        while waypoint_index < len(waypoints):
            tx, ty = waypoints[waypoint_index]
            if math.hypot(tx - x, ty - y) < waypoint_tolerance:
                waypoint_index += 1
                continue
            break

        if waypoint_index >= len(waypoints):
            break

        tx, ty = waypoints[waypoint_index]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_err = DifferentialDriveRobot._wrap_angle(target_heading - theta)

        omega = max(-max_omega, min(max_omega, 4.0 * heading_err))
        if abs(heading_err) > 0.45:
            v = 0.0
        else:
            forward_alignment = max(0.0, math.cos(heading_err))
            distance_scale = min(1.0, dist / 40.0)
            v = max_v * forward_alignment * distance_scale

        commands.append((v, omega))

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = DifferentialDriveRobot._wrap_angle(theta + omega * dt)

    commands.append((0.0, 0.0))
    return commands


@dataclass(slots=True)
class SimState:
    world: GridMap
    robot: DifferentialDriveRobot
    trajectory_optimizer: TrajectoryOptimizer
    mode: ControlMode = ControlMode.TeleOp
    goal: tuple[float, float] | None = None
    path_points: list[tuple[float, float]] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    waypoint_index: int = 0
    trajectory: list[tuple[float, float]] | None = None

    def set_goal(self, x: float, y: float) -> None:
        if self.world.is_occupied(x, y):
            return
        gx, gy = self.world.world_to_grid(x, y)
        self.goal = self.world.grid_to_world_center(gx, gy)
        self.path_points = self.world.astar((self.robot.x, self.robot.y), self.goal)
        self.waypoints = self.path_points.copy()
        self.waypoint_index = 0

    def update_autonav(self) -> None:
        if not self.waypoints or self.waypoint_index >= len(self.waypoints):
            self.robot.v = 0.0
            self.robot.omega = 0.0
            return

        # Consume waypoints aggressively so the robot does not orbit a point.
        tx, ty = self.waypoints[self.waypoint_index]
        dx, dy = tx - self.robot.x, ty - self.robot.y
        dist = math.hypot(dx, dy)
        while dist < 10.0 and self.waypoint_index < len(self.waypoints) - 1:
            self.waypoint_index += 1
            tx, ty = self.waypoints[self.waypoint_index]
            dx, dy = tx - self.robot.x, ty - self.robot.y
            dist = math.hypot(dx, dy)

        if dist < 10.0 and self.waypoint_index == len(self.waypoints) - 1:
            self.robot.v = 0.0
            self.robot.omega = 0.0
            return

        target_heading = math.atan2(dy, dx)
        heading_err = DifferentialDriveRobot._wrap_angle(target_heading - self.robot.theta)
        self.robot.omega = max(-self.robot.max_omega, min(self.robot.max_omega, 4.0 * heading_err))

        # Stop-turn-go controller: rotate in place for large heading errors.
        if abs(heading_err) > 0.45:
            self.robot.v = 0.0
            return

        forward_alignment = max(0.0, math.cos(heading_err))
        distance_scale = min(1.0, dist / 40.0)
        self.robot.v = self.robot.max_v * forward_alignment * distance_scale

    def update_traj_opt(self) -> None:
        if self.goal is None:
            self.robot.v = 0.0
            self.robot.omega = 0.0
            return
        
        goal_x, goal_y = self.goal

        distance = math.sqrt((goal_x - self.robot.x) ** 2 + (goal_y - self.robot.y) ** 2)
        if distance < 0.1 * CELL_SIZE:
            self.robot.v = 0.0
            self.robot.omega = 0.0
            return
        
        path_points = self.world.astar((self.robot.x, self.robot.y), self.goal)
        initial_commands = waypoints_to_commands(
            path_points,
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            1.0 / FPS,
            self.robot.max_v,
            self.robot.max_omega,
            max_steps=60)
        (controls_v, controls_omega, x_traj, y_traj) = self.trajectory_optimizer.find_path(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            goal_x,
            goal_y,
            initial_commands=initial_commands)
        
        self.robot.v = controls_v[0].item()
        self.robot.omega = controls_omega[0].item()
        self.trajectory = []
        for i in range(x_traj.shape[0]):
            self.trajectory.append((x_traj[i].item(), y_traj[i].item()))


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Trajectory Optimization Sandbox")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)

    world = make_demo_map()
    rx, ry = world.grid_to_world_center(2, 2)
    robot = DifferentialDriveRobot(rx, ry, theta=0.0)
    state = SimState(
        world=world,
        robot=robot,
        trajectory_optimizer=TrajectoryOptimizer(world, dt=2.0 * (1.0/FPS), max_v=robot.max_v, max_omega=robot.max_omega))

    running = True
    while running:
        clock.tick(FPS)
        dt = 1.0 / FPS
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    state.mode = state.mode.next()
                    state.trajectory = None
                    if state.mode == ControlMode.TeleOp:
                        state.robot.v = 0.0
                        state.robot.omega = 0.0
                elif event.key == pygame.K_r:
                    state.robot.trail.clear()
                    state.waypoints.clear()
                    state.path_points.clear()
                    state.goal = None
                    state.waypoint_index = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                state.set_goal(mx, my)

        if state.mode == ControlMode.TeleOp:
            v_cmd = 0.0
            omega_cmd = 0.0
            if keys[pygame.K_UP]:
                v_cmd += state.robot.max_v
            if keys[pygame.K_DOWN]:
                v_cmd -= state.robot.max_v
            if keys[pygame.K_LEFT]:
                omega_cmd -= state.robot.max_omega
            if keys[pygame.K_RIGHT]:
                omega_cmd += state.robot.max_omega
            state.robot.v = v_cmd
            state.robot.omega = omega_cmd
        elif state.mode == ControlMode.AutoNav:
            state.update_autonav()
        elif state.mode == ControlMode.TrajOptim:
            state.update_traj_opt()

        state.robot.step(dt, state.world)

        screen.fill(BG_COLOR)

        for y in range(state.world.height):
            for x in range(state.world.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cx, cy = state.world.grid_to_world_center(x, y)
                color = OBSTACLE_COLOR if state.world.is_occupied(cx, cy) else GRID_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BG_COLOR, rect, width=1)

        if len(state.robot.trail) > 1:
            pygame.draw.lines(screen, (105, 160, 255), False, state.robot.trail, width=2)

        if state.trajectory is not None:
            pygame.draw.lines(screen, (255, 160, 105), False, state.trajectory, width=2)

        if state.path_points and state.mode == ControlMode.AutoNav:
            points = state.path_points
            if len(points) > 1:
                pygame.draw.lines(screen, PATH_COLOR, False, points, width=3)
            for p in points:
                pygame.draw.circle(screen, PATH_COLOR, (int(p[0]), int(p[1])), 3)

        pygame.draw.circle(screen, ROBOT_COLOR, (int(state.robot.x), int(state.robot.y)), int(state.robot.radius))
        hx = state.robot.x + math.cos(state.robot.theta) * state.robot.radius
        hy = state.robot.y + math.sin(state.robot.theta) * state.robot.radius
        pygame.draw.line(
            screen,
            HEADING_COLOR,
            (int(state.robot.x), int(state.robot.y)),
            (int(hx), int(hy)),
            width=3,
        )

        if state.goal is not None:
            pygame.draw.circle(screen, (235, 78, 107), (int(state.goal[0]), int(state.goal[1])), 8, width=2)

        mode_text = f"mode: {state.mode.name}"
        controls = "arrows: drive  |  left-click: goal  |  space: toggle auto  |  r: clear path"
        screen.blit(font.render(mode_text, True, TEXT_COLOR), (10, 8))
        screen.blit(font.render(controls, True, TEXT_COLOR), (10, WINDOW_H - 28))

        pygame.display.flip()

    pygame.quit()
