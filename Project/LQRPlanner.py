import numpy as np
import math
import control
from scipy.linalg import expm

class LQRPlanner:
    def __init__(self, map_data, A, B, C):
        self.map = map_data
        self.start = None
        self.goal = None
        self.obstacles = None # obstacles stored as matrix where columns represent each obstacle location

        # dynamics
        self.A = A
        self.B = B
        self.C = C

        self.S = None
        self.L = None
        self.E = None
        self.Q = np.identity(2)
        self.R = np.identity(2)

        self.A_tilda = None
        self.B_tilda = None

        # LQRObstacles parameters
        self.euclid_tolerance = 0.15# configuration tolerance within which 2 configurations are considered equal
        self.time_ub = 1 # upper bound for LQRObstacles time iterations
        self.dt = 0.1

        print("initing control matrices")
        self.init_matrices()
        self.init_obstacles()

        print("initialized control matrices")

    def add_obs(self, obs):
        self.obstacles = np.hstack((self.obstacles, obs))
        print("self.obstacles is ", self.obstacles)

    def init_obstacles(self):
        obstacles = []
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if (self.map[i][j] == 1):
                    obstacles.append([j, i])
        self.obstacles = np.array(obstacles).T

    def init_matrices(self):
        self.S = control.care(self.A, self.B, self.C.T @ self.Q @ self.C, R=self.R)[0]
        self.L = np.linalg.inv(self.R) @ self.B.T @ self.S
        self.E = np.linalg.inv(self.R) @ self.B.T @ np.linalg.inv((self.B @ self.L) - self.A).T @ self.C.T @ self.Q

        self.A_tilda = self.A - self.B @ self.L
        self.B_tilda = self.B @ self.E

    def euclid(self, v1, v2):
        return math.sqrt((v1[0]-v2[0])**2 + (v1[1] - v2[1])**2)

    def LQR_obstacles(self, x):
        """ x should be a numpy array of size (n,1)"""
        tolerance = 0.000001
        invalid_configs = []
        #linspace = np.linspace(0, 20, 40)[1:]
        #for t in np.linspace(0, 60, 80)[1:]:
        for t in np.linspace(0, 20, 40)[1:]:
            F_t = expm(t * self.A_tilda)
            G_t = np.linalg.inv(self.A_tilda) @ (expm(t * self.A_tilda) - np.identity(4)) @ self.B_tilda

            O_trans = np.linalg.inv(self.C @ G_t) @ (self.obstacles + (-self.C @ F_t @ x))

            for j in range(O_trans.shape[1]):
                config = O_trans[:,j]
                found = False
                for vec in invalid_configs:
                    if (self.euclid(config, vec) < tolerance):
                        found = True
                        break
                if found: continue
                invalid_configs.append(config)
        return invalid_configs

    def LQR_obstacles_test(self, x, obs):
        """ x should be a numpy array of size (n,1)"""
        tolerance = 0.01
        invalid_configs = []
        #linspace = np.linspace(0, 20, 40)[1:]
        for t in np.linspace(0, 20, 40)[1:]:
        #for t in np.append(np.linspace(0, self.time_ub, int(self.time_ub/self.dt))[1:], [2000]):
            F_t = expm(t * self.A_tilda)
            G_t = np.linalg.inv(self.A_tilda) @ (expm(t * self.A_tilda) - np.identity(4)) @ self.B_tilda

            O_trans = np.linalg.inv(self.C @ G_t) @ (obs + (-self.C @ F_t @ x))

            for j in range(O_trans.shape[1]):
                config = O_trans[:,j]
                found = False
                for vec in invalid_configs:
                    if (self.euclid(config, vec) < tolerance):
                        found = True
                        break
                if found: continue
                invalid_configs.append(config)
        return invalid_configs

    def stringify(self, pos):
        return str(pos[0]) + "&" + str(pos[1])

    def tupify(self, pos_str):
        coords = pos_str.split("&")
        return (int(coords[0]), int(coords[1]))

    def within_bounds(self, loc):
        x, y = loc[0], loc[1]

        return (x >= 0) and (x < self.map.shape[1]) and (y >= 0) and (y < self.map.shape[0])

    def extract_path(self, visited, goal):
        path = [goal]
        parent_key = visited[self.stringify(goal)]

        while (parent_key is not None):
            path.insert(0, self.tupify(parent_key))
            parent_key = visited[parent_key]
        return path

    def get_children(self, pos):
        dxs = [-1, -1, 0, 1, 1, 1, 0, -1]
        dys = [0, -1, -1, -1, 0, 1, 1, 1]

        children = []
        x, y = pos
        lqr_obstacles = self.LQR_obstacles(np.array([x, y, 0, 0]).reshape((4,1)))
        for i in range(len(dxs)):
            if (not self.within_bounds((x + dxs[i], y + dys[i]))): continue
            loc = np.array([x + dxs[i], y + dys[i]]).reshape((2,1))
            if (self.check_collision_free_control(loc, lqr_obstacles)):
                if ((x == 8) and (y == 4)):
                    if (loc[0][0] == 8) and (loc[1][0] == 5):
                        print("passing 8,5 as valid child to 8,4")
                        closest_obs = None
                        min_dist = 99999999999
                        for obstacle in lqr_obstacles:
                            if (self.euclid(loc, obstacle) < min_dist):
                                min_dist = self.euclid(loc, obstacle)
                                closest_obs = obstacle
                        print("min_dist is ", min_dist)
                        print("closest_obs is ", closest_obs)
                children.append((x + dxs[i], y + dys[i]))
        return children

    def get_children_basic(self, pos):
        dxs = [-1, -1, 0, 1, 1, 1, 0, -1]
        dys = [0, -1, -1, -1, 0, 1, 1, 1]

        children = []
        x, y = pos
        for i in range(len(dxs)):
            if (not self.within_bounds((x + dxs[i], y + dys[i]))): continue
            if (self.map[y + dys[i]][x + dxs[i]] == 0):
                children.append((x + dxs[i], y + dys[i]))
        return children

    def check_collision_free_control(self, loc, obstacles):
        for obstacle in obstacles:
            if (self.euclid(loc, obstacle) < self.euclid_tolerance):
                return False
        return True

    def search(self, start, goal):
        """ run BFS search """
        visited = dict()
        visited[self.stringify(start)] = None

        horizon = [start]

        while (len(horizon) > 0):
            position = horizon.pop(0)
            if (position == goal):
                return self.extract_path(visited, goal)

            for child in self.get_children(position):
                if (self.stringify(child) in visited): continue
                visited[self.stringify(child)] = self.stringify(position)
                horizon.append(child)
        return None

    def search_basic(self, start, goal):
        """ run BFS search """
        visited = dict()
        visited[self.stringify(start)] = None

        horizon = [start]

        while (len(horizon) > 0):
            position = horizon.pop(0)
            if (position == goal):
                return self.extract_path(visited, goal)

            for child in self.get_children_basic(position):
                if (self.stringify(child) in visited): continue
                visited[self.stringify(child)] = self.stringify(position)
                horizon.append(child)
        return None

    def track(self, traj):
        path = [traj[0]]
        target_ind = 1
        t = 0
        dt = 0.2
        x = np.array([traj[0][0], traj[0][1], 0, 0]).reshape((4,1))
        tolerance = 0.001
        while (target_ind < len(traj)):
            c = np.array([traj[target_ind][0], traj[target_ind][1]]).reshape((2,1))
            u = (-self.L @ x) + self.E @ c
            x_dot = (self.A @ x) + (self.B @ u)
            x = x + x_dot * dt
            path.append((x[0][0], x[1][0]))
            t += dt
            target_ind = math.ceil(t)
            if (target_ind == len(traj)) and (self.euclid((x[0][0], x[1][0]), c) > tolerance):
                target_ind = len(traj)-1
        return path

    def get_best_target(self, goal, x):
        lqr_obstacles = self.LQR_obstacles(x)
        if (self.check_collision_free_control(goal, lqr_obstacles)):
            return [goal[0], goal[1]]

        # get candidate locations
        candidates = []
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if (self.map[i][j] == 0) and (self.check_collision_free_control((j, i), lqr_obstacles)):
                    candidates.append((j, i))
        min_dist = 9999999999999
        best_candidate = None 
        for candidate in candidates:
            dist = self.euclid(candidate, goal)
            if (dist < min_dist):
                min_dist = dist
                best_candidate = candidate
        return [best_candidate[0], best_candidate[1]]

    def approach(self, start, goal):
        path = [start]
        dt = 0.2
        tolerance = 0.001
        x = np.array([start[0], start[1], 0, 0]).reshape((4,1))
        c = np.array(self.get_best_target(goal, x)).reshape((2,1))
        while(self.euclid((x[0][0], x[1][0]), c) > tolerance):
            u = (-self.L @ x) + self.E @ c
            x_dot = (self.A @ x) + (self.B @ u)
            x = x + x_dot * dt
            path.append((x[0][0], x[1][0]))
            c = np.array(self.get_best_target(goal, x)).reshape((2,1))
        return path