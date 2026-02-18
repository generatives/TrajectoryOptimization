import math
import torch
import time

class TrajectoryOptimizer:
    def __init__(self,
                 map,
                 num_steps: int = 60,
                 dt: float = 1./60.,
                 max_v: float = 150.0,
                 max_omega: float = 3.0,):
        self.num_steps = num_steps
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.map = map
        self.obstacle_centers = torch.tensor(list(map.get_occupied_centers()))
        self.potential_distance = self.max_v * self.dt * self.num_steps

        half_cell_size = map.cell_size / 2.0
        cell_radius = math.sqrt(half_cell_size ** 2 + half_cell_size ** 2)
        self.obstacle_radius = torch.tensor([cell_radius])

    def _robot_model(self, x, y, theta, command_v, command_omega, dt):
        nx = x + command_v * torch.cos(theta) * dt
        ny = y + command_v * torch.sin(theta) * dt
        ntheta = theta + command_omega * dt
        return (nx, ny, ntheta)
    
    @torch.compile
    def _model(self, x, y, theta, commands_v, commands_omega, dt):
        x_traj = torch.zeros_like(commands_v)
        y_traj = torch.zeros_like(commands_v)
        theta_traj = torch.zeros_like(commands_v)

        commands_v = torch.clamp(commands_v, min=-self.max_v, max=self.max_v)
        commands_omega = torch.clamp(commands_omega, min=-self.max_omega, max=self.max_omega)

        for i in range(self.num_steps):
            (x, y, theta) = self._robot_model(x, y, theta, commands_v[i], commands_omega[i], dt)
            x_traj[i] = x
            y_traj[i] = y
            theta_traj[i] = theta
        return (x_traj, y_traj, theta_traj)
    
    def _distance_to_goal(self, x, y, goal_x, goal_y):
        return (goal_x - x) ** 2 + (goal_y - y) ** 2
    
    def _obstacle_cost(self, x_traj, y_traj, relevant_obstacles):
        trajectory = torch.stack([x_traj, y_traj], dim=1)
        distances = torch.cdist(trajectory, self.obstacle_centers).flatten()
        costs = torch.log(1 + torch.e ** (self.obstacle_radius - distances))
        return torch.sum(costs)
    
    @torch.compile
    def _loss(self, x_traj, y_traj, relevant_obstacles, goal_x, goal_y, commands_v, commands_omega):
        distance = self._distance_to_goal(x_traj[self.num_steps - 1], y_traj[self.num_steps - 1], goal_x, goal_y)
        velocity_cost = (0.001 / self.num_steps) * torch.sum(torch.sqrt(commands_v ** 2))
        omega_cost = (0.001 / self.num_steps) * torch.sum(torch.sqrt(commands_omega ** 2))
        obstacle_cost = 10.0 * self._obstacle_cost(x_traj, y_traj, relevant_obstacles)
        return distance \
            + obstacle_cost
            #+ velocity_cost \
            #+ omega_cost \
    
    def find_path(self, x, y, theta, goal_x, goal_y, initial_commands=None):

        if initial_commands is None:
            distance = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

            init_velocity_steps = int(min((distance / self.potential_distance), 1.0) * self.num_steps)

            commands_v = torch.randn(self.num_steps)
            commands_v[:init_velocity_steps] = self.max_v
            commands_v.requires_grad_()
            commands_omega = torch.randn(self.num_steps)
            commands_omega.requires_grad_()
        else:
            commands_v = torch.tensor([cmd[0] for cmd in initial_commands[:self.num_steps]])
            commands_v.requires_grad_()
            commands_omega = torch.tensor([cmd[1] for cmd in initial_commands[:self.num_steps]])
            commands_omega.requires_grad_()


        optimizer = torch.optim.Adam([commands_v, commands_omega], lr=0.9)
        
        tensor_x = torch.tensor([x])
        tensor_y = torch.tensor([y])
        position = torch.tensor([x, y])
        tensor_theta = torch.tensor([theta])
    
        last_loss = torch.tensor([torch.inf])
        change = torch.tensor([torch.inf])
        iterations = 0

        start_time = time.time()
        while iterations < 100:
            optimizer.zero_grad()
            (x_traj, y_traj, _) = self._model(tensor_x, tensor_y, tensor_theta, commands_v, commands_omega, self.dt)
            obstacle_distances = torch.dist(position, self.obstacle_centers)
            relevant_obstacles = self.obstacle_centers[obstacle_distances < self.potential_distance]
            loss = self._loss(x_traj, y_traj, relevant_obstacles, goal_x, goal_y, commands_v, commands_omega)
            loss.backward()
            optimizer.step()
            change = loss - last_loss
            last_loss = loss
            iterations += 1
        end_time = time.time()
        duration = end_time - start_time

        #print(f"Done after {iterations} iterations with change {change}. Final loss {last_loss}. Duration: {duration}")
        return (commands_v.detach().numpy(), commands_omega.detach().numpy(), x_traj.detach().numpy(), y_traj.detach().numpy())



if __name__ == "__main__":
    optimizer = TrajectoryOptimizer()
    controls = optimizer.find_path(80.0, 80.0, 0.0, 1000.0, 80.0)
    print(controls)