import math
import torch
import time

torch.autograd.set_detect_anomaly(True)

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
        self.build_collision_maps(map)
        self.potential_distance = self.max_v * self.dt * self.num_steps

        half_cell_size = map.cell_size / 2.0
        cell_radius = math.sqrt(half_cell_size ** 2 + half_cell_size ** 2)
        self.obstacle_radius = torch.tensor([cell_radius])

    def build_collision_maps(self, map):
        self.occupied_west = torch.zeros((map.width, map.height), dtype=torch.bool)
        self.occupied_east = torch.zeros((map.width, map.height), dtype=torch.bool)
        self.occupied_north = torch.zeros((map.width, map.height), dtype=torch.bool)
        self.occupied_south = torch.zeros((map.width, map.height), dtype=torch.bool)
        for x in range(map.width):
            for y in range(map.height):
                if map.is_occupied(x - 1, y):
                    self.occupied_west[x, y] = True
                
                if map.is_occupied(x + 1, y):
                    self.occupied_east[x, y] = True
                    
                if map.is_occupied(x, y - 1):
                    self.occupied_north[x, y] = True
                    
                if map.is_occupied(x, y + 1):
                    self.occupied_south[x, y] = True

    def velocity_scale(self, distance):
        return 1 / (1 + torch.exp(-(distance - 5)))

    def collision_check(self, nx, ny, theta, command_v):
        velocity = command_v * torch.stack([torch.cos(theta), torch.sin(theta)])
        grid_x = (nx / self.map.cell_size).int()
        grid_y = (ny / self.map.cell_size).int()

        west_distance = nx - (grid_x * self.map.cell_size).detach()
        west_scale = torch.stack([self.velocity_scale(west_distance), torch.ones((1,))])

        east_distance = ((grid_x+1) * self.map.cell_size).detach() - nx
        east_scale = torch.stack([self.velocity_scale(east_distance), torch.ones((1,))])

        north_distance = ny - (grid_y * self.map.cell_size).detach()
        north_scale = torch.stack([torch.ones((1,)), self.velocity_scale(north_distance)])

        south_distance = ((grid_y+1) * self.map.cell_size).detach() - ny
        south_scale = torch.stack([torch.ones((1,)), self.velocity_scale(south_distance)])

        colliding_west = self.occupied_west[grid_x, grid_y] & (velocity[0] < 0)
        velocity = torch.where(colliding_west, velocity * west_scale, velocity)

        colliding_east = self.occupied_east[grid_x, grid_y] & (velocity[0] > 0)
        velocity = torch.where(colliding_east, velocity * east_scale, velocity)

        colliding_north = self.occupied_north[grid_x, grid_y] & (velocity[1] < 0)
        velocity = torch.where(colliding_north, velocity * north_scale, velocity)

        colliding_south = self.occupied_south[grid_x, grid_y] & (velocity[1] > 0)
        velocity = torch.where(colliding_south, velocity * south_scale, velocity)

        return velocity

    def _robot_model(self, x, y, theta, command_v, command_omega, dt):
        nx = x + command_v * torch.cos(theta) * dt
        ny = y + command_v * torch.sin(theta) * dt
        velocity = self.collision_check(nx, ny, theta, command_v)
        nx = x + velocity[0] * dt
        ny = y + velocity[1] * dt
        ntheta = theta + command_omega * dt
        return (nx, ny, ntheta)
    
    #@torch.compile
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
    
    #@torch.compile
    def _loss(self, x_traj, y_traj, relevant_obstacles, goal_x, goal_y, commands_v, commands_omega):
        distance = self._distance_to_goal(x_traj[self.num_steps - 1], y_traj[self.num_steps - 1], goal_x, goal_y)
        velocity_cost = (0.001 / self.num_steps) * torch.sum(torch.sqrt(commands_v ** 2))
        omega_cost = (0.001 / self.num_steps) * torch.sum(torch.sqrt(commands_omega ** 2))
        #obstacle_cost = 10.0 * self._obstacle_cost(x_traj, y_traj, relevant_obstacles)
        return distance \
            #+ obstacle_cost
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
        while iterations < 1:
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