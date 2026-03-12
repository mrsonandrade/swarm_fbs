# -*- coding: utf-8 -*-

import numpy as np
import random
import math
from shapely.geometry import LineString, Point
from scipy.signal import savgol_filter
from pso_optimizer import objective, get_positions, positions_list
from pyswarm import pso
from scipy.optimize import linear_sum_assignment

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def build_path(path_xy):
    line = LineString(path_xy)
    seg_lengths = np.sqrt(np.sum(np.diff(path_xy, axis=0)**2, axis=1))
    cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
    return line, cumlen

def project_to_path(x, y, line, cumlen, path_xy):
    p = Point(x, y)

    dist_along = line.project(p)
    nearest_point = line.interpolate(dist_along)
    px, py = nearest_point.x, nearest_point.y

    seg_idx = np.searchsorted(cumlen, dist_along) - 1
    seg_idx = np.clip(seg_idx, 0, len(path_xy) - 2)

    p1 = path_xy[seg_idx]
    p2 = path_xy[seg_idx + 1]
    tangent = p2 - p1
    tangent = tangent / np.linalg.norm(tangent)
    tx, ty = tangent

    nx, ny = -ty, tx  # left-normal
    cross_track = (x - px) * nx + (y - py) * ny

    path_heading = math.atan2(ty, tx)

    return {
        "px": px,
        "py": py,
        "cross_track": cross_track,
        "path_heading": path_heading,
        "progress": dist_along,
    }


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=1.0, goal_sample_rate=10, max_iter=1000):
        self.start = Node(*start)
        self.end = Node(*goal)
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

    def planning(self):
        for _ in range(self.max_iter):
            rnd = self.sample_free()
            nearest_ind = self.get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]

            theta = np.arctan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = Node(nearest_node.x + self.expand_dis * np.cos(theta),
                            nearest_node.y + self.expand_dis * np.sin(theta))
            new_node.parent = nearest_node

            if not self.collision_check(new_node):
                continue

            self.node_list.append(new_node)

            if np.hypot(new_node.x - self.end.x, new_node.y - self.end.y) < self.expand_dis:
                return self.extract_path(new_node)

        return None

    def sample_free(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            return [np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(self.min_rand, self.max_rand)]
        return [self.end.x, self.end.y]

    def get_nearest_node_index(self, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in self.node_list]
        return int(np.argmin(dlist))

    def collision_check(self, node):
        for (ox, oy, size) in self.obstacle_list:
            if (ox - node.x)**2 + (oy - node.y)**2 <= size**2:
                return False
        return True

    def extract_path(self, goal_node):
        path = [[goal_node.x, goal_node.y]]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1]

def hungarian_algorithm(initial_positions, final_positions):
    n = len(initial_positions)
    cost_matrix = np.zeros((n, n))
    for i, (x1, y1) in enumerate(initial_positions):
        for j, (x2, y2) in enumerate(final_positions):
            cost_matrix[i, j] = np.linalg.norm([x1 - x2, y1 - y2])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_cost = cost_matrix[row_ind, col_ind].sum()

    bw_initial = []
    bw_final = []
    for i, j in zip(row_ind, col_ind):
        bw_initial.append(i)
        bw_final.append(j)

    return bw_initial, bw_final, total_cost

class Vessel(object):
    def __init__(self,
                 max_steps):

        # main characteristics
        self.g = 9.8 # gravity acc
        self.L = 38. # vessel length (meters)
        self.B = 16. # vessel beam
        self.D = 16. # depth
        self.T = 8. # draft
        self.rho = 1025. # water density kg/m3
        self.mass = self.rho * (self.L*self.B*self.T)
        self.Ixx = (self.mass/12.)*(self.B**2 + self.D**2)  # Moment of inertia
        self.Iyy = (self.mass/12.)*(self.B**2 + self.L**2)
        self.Izz = (self.mass/12.)*(self.D**2 + self.L**2)
        self.dt = 0.5

        self.world_x_min = -500  # meters
        self.world_x_max =  500
        self.world_y_min = -500
        self.world_y_max =  500

        self.initial_bw_positions = []
        self.other_bw_positions = []
        self.obstacle_list_bw = []

        self.target_x, self.target_y, self.target_r = 0, 0, 19.

        self.already_achieved = False
        self.max_steps = max_steps

        self.step_id = 0

        self.dist_goal0 = 0
        self.goal = []
        self.obstacle_list = []
        self.rrt = None
        self.path = None
        self.dist_path = 0
        self.prev_progress = 1.
        self.fopt = None
        self.x_offset, self.y_offset = 0, 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        self.state_dims = 10
        self.action_dims = len(self.action_table)

        self.X_u, self.Y_v, self.N_r, self.X_udot, self.Y_vdot, self.N_rdot = self.hydrodynamic_coefficients(self.rho, self.L, self.B, self.T)

        self.state_buffer = []

    def hydrodynamic_coefficients(self, rho, L, B, T):
        C_Dx = 1.0
        C_Dy = 1.8
        C_Dn = 1.0
        U_ref = 1.0
        V_ref = 1.0
        r_ref = 0.01

        A_surge = B * T
        A_sway = L * T
        X_u = -0.5 * rho * C_Dx * A_surge * U_ref
        Y_v = -0.5 * rho * C_Dy * A_sway * V_ref
        N_r = -0.5 * rho * C_Dn * A_sway * (L ** 2 / 12) * r_ref

        C_a = np.pi / 4.
        X_udot = -rho * C_a * T * B * 0.05
        Y_vdot = -rho * C_a * L * B ** 2
        N_rdot = -rho * C_a * B ** 2 * (L ** 3 / 12)

        return X_u, Y_v, N_r, X_udot, Y_vdot, N_rdot

    def smooth_path_savgol(self, path, window=30, poly=3, out_points=50):
        path = np.asarray(path)

        x = savgol_filter(path[:, 0], window, poly, mode='nearest')
        y = savgol_filter(path[:, 1], window, poly, mode='nearest')
        smoothed = np.column_stack((x, y))

        idx = np.linspace(0, len(smoothed) - 1, out_points).astype(int)
        idx = idx.tolist()
        idx.insert(0, 0)
        idx.append(len(smoothed) - 1)
        idx = np.asarray(idx)

        smoothed = smoothed[idx].tolist()
        smoothed.insert(0, path[0])
        smoothed.append(path[-1])

        return np.asarray(smoothed)

    def reset(self, max_attempts=10):
        for _ in range(max_attempts):

            self.state = self.create_random_state()
            self.state_buffer = []
            self.step_id = 0
            self.already_achieved = False

            self.goal = np.array([0.0, 0.0])
            xy = np.array([self.state['x'], self.state['y']], dtype=float)

            self.rrt = RRT(
                start=xy,
                goal=self.goal,
                obstacle_list=self.obstacle_list_bw,
                rand_area=(-500, 500),
                expand_dis=2.,
                max_iter=10000,
                goal_sample_rate=10
            )

            path = self.rrt.planning()

            if path is None:
                continue

            path = np.asarray(path, dtype=float)

            if len(path) < 5:
                continue

            self.path = self.smooth_path_savgol(path)
            self.prev_progress = 0.0

            return self.flatten(self.state)

        self.path = np.array([[self.state['x'], self.state['y']], [0.0, 0.0]])
        self.prev_progress = 0.0
        return self.flatten(self.state)

    def create_action_table(self):
        forces = [-1000000, 1000000]  # N
        moments = [0]  # N·m (yaw)
        action_table = [[X, Y, N] for X in forces for Y in forces for N in moments]
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):

        valid_random_state = False

        def get_random_state():
            last_theta =  random.randint(180, 225)
            last_T = random.randint(1, 11)
            # PSO bounds (example)
            lb = np.array([1] * 10, dtype=float)
            ub = np.array([26] * 10, dtype=float)
            args = (last_T, last_theta)
            # Run PSO (this can take time)
            xopt, fopt = pso(objective, lb, ub,
                             args=args, swarmsize=20, omega=0.5,
                             phip=1.9, phig=2.5, maxiter=250,
                             minstep=1e-8, minfunc=1e-8, debug=False)
            xopt2 = np.round(xopt).astype(int)
            return xopt2, fopt

        initial_positions_index = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
        initial_state = None
        final_state = None
        n = None

        while valid_random_state == False:

            xopt2, fopt = get_random_state()
            self.fopt = fopt

            final_position_index = xopt2
            initial_positions = get_positions(np.asarray(positions_list), initial_positions_index).T
            final_positions = get_positions(np.asarray(positions_list), final_position_index).T
            bw_initial, bw_final, total_cost = hungarian_algorithm(initial_positions, final_positions)
            initial_state = initial_positions[bw_initial]
            final_state = final_positions[bw_final]

            valid_indices = [i for i in range(len(initial_state)) if (final_state[i] != initial_state[i]).all()]

            if 2 in valid_indices:
                n = 2
                valid_random_state = True

        x0, y0 = initial_state[n]
        tgt_x, tgt_y = final_state[n]

        self.x_offset, self.y_offset = tgt_x, tgt_y

        tgt = np.asarray([tgt_x, tgt_y])

        self.other_bw_positions = final_state
        self.other_bw_positions = self.other_bw_positions - tgt
        self.initial_bw_positions = initial_state - tgt

        self.obstacle_list_bw = []
        for pi in np.delete(self.other_bw_positions, n, axis=0):
            self.obstacle_list_bw.append([pi[0], pi[1], np.sqrt(3.)*19])

        x = x0 - tgt_x
        y = y0 - tgt_y
        psi = 0.5 * np.pi

        state = {
            'x': x,
            'y': y,
            'psi': psi,
            'u': 0,
            'v': 0,
            'r': 0,
            'dist_target': 0,
            'dist_pathx':0 ,
            'dist_pathy': 0,
            't': 0,
            'a_': 0
        }

        return state

    def check_navigating_success(self, state):
        x, y, psi = state['x'], state['y'], state['psi']
        if np.linalg.norm([x, y])<19.:
            return True
        else:
            return False

    def check_speed(self, u, v):
        if abs(u)>2.0 and abs(v)>2.0:
            return 2.0
        elif abs(u)>2.0 or abs(v)>2.0:
            return 1.0
        else:
            return -1.0

    def reward_dense(
            self,
            state,
            action,
            t,
            next_state,
            line,
            cumlen,
            path_xy,
            prev_progress,
            goal_point=(0.0, 0.0),
            reach_thresh=5.0,
            w_progress=1.0,
            w_cross=1.0,
            w_heading=0.2,
            w_act=0.01,
            w_time=0.001,
            w_speed=1.0
    ):

        x, y, theta, u, v = state[:5]
        nx, ny, ntheta = next_state[:3]

        info = project_to_path(nx, ny, line, cumlen, path_xy)

        cross_track = info["cross_track"]
        path_heading = info["path_heading"]
        progress = info["progress"]
        delta_progress = progress - prev_progress

        delta_progress = max(delta_progress, -0.05)

        heading_err = wrap_angle(ntheta - path_heading)

        r = 0.0
        r += w_progress * delta_progress

        if abs(cross_track)>1.0:
            r -= w_cross * abs(cross_track)
        elif abs(cross_track)<=1.0:
            r += w_cross * abs(cross_track)

        r -= w_speed * self.check_speed(u, v)
        r -= w_time

        if math.hypot(nx - goal_point[0], ny - goal_point[1]) <= reach_thresh:
            r += 1000.0
            done = True
        else:
            done = False

        return r, progress, done, cross_track, delta_progress

    def check_time_and_distance(self, state):
        t, x, y, psi = state['t'], state['x'], state['y'], state['psi']
        percentage_path = (np.linalg.norm([x, y])/self.dist_goal0)
        if   (t > 200 ) and (percentage_path > 0.8):
            return True
        else:
            return False

    def calculate_reward(self, state, action, next_state):
        line, cumlen = build_path(self.path)
        total_reward, new_progress, done, cross_track, delta_progress = self.reward_dense(np.array([state['x'], state['y'], state['psi'], state['u'], state['v']], dtype=float),
                                    action,
                                    state['t'],
                                    next_state,
                                    line,
                                    cumlen,
                                    self.path,
                                    self.prev_progress,
                                    goal_point=(0.0, 0.0),
                                    reach_thresh=19.00,
                                    w_progress=2.5,
                                    w_cross=0.25,
                                    w_heading=0.25,
                                    w_act=0.025,
                                    w_time=0.5,
                                    w_speed=2.5)

        self.prev_progress = new_progress

        return total_reward, new_progress, cross_track, delta_progress

    def step(self, action):
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        u, v, r = self.state['u'], self.state['v'], self.state['r']

        tau = np.array(self.action_table[action])

        m = self.mass
        Iz = self.Izz
        X_u = self.X_u
        Y_v = self.Y_v
        N_r = self.N_r
        X_udot = self.X_udot
        Y_vdot = self.Y_vdot
        N_rdot = self.N_rdot

        M_RB = np.diag([m, m, Iz])

        M_A = -np.diag([X_udot, Y_vdot, N_rdot])

        M = M_RB + M_A

        C_RB = np.array([[0, -m * r, 0],
                      [m * r, 0, 0],
                      [0, 0, 0]])

        C_A = np.zeros((3, 3))

        C = C_RB + C_A

        D = np.diag([-X_u, -Y_v, -N_r])

        nu = np.array([u, v, r])
        nu_dot = np.linalg.inv(M) @ (tau - (C + D) @ nu)

        nu_new = nu + nu_dot * self.dt
        u_new, v_new, r_new = nu_new

        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
        eta_dot = R @ nu
        eta_new = np.array([x, y, psi]) + eta_dot * self.dt
        x_new, y_new, psi_new = eta_new

        self.state = {
            'x': x_new,
            'y': y_new,
            'psi': psi_new,
            'u': u_new,
            'v': v_new,
            'r': r_new,
            'dist_target': 0,
            'dist_pathx':0 ,
            'dist_pathy': 0,
            't': self.step_id,
            'action_': action
        }
        self.step_id += 1

        done = False
        next_state = np.array([x_new, y_new, psi_new], dtype=float)
        reward, dist_target, dist_pathx, dist_pathy  = self.calculate_reward(self.state, action, next_state)
        self.state['dist_target'] = dist_target
        self.state['dist_pathx'] = dist_pathx
        self.state['dist_pathy'] = dist_pathy
        self.state_buffer.append(self.state)
        self.already_achieved = self.check_navigating_success(self.state)

        if self.already_achieved:
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, None

    def flatten(self, state):
        x = [state['x'],
             state['y'],
             state['psi'],
             state['u'],
             state['v'],
             state['r'],
             state['dist_target'],
             state['dist_pathx'],
             state['dist_pathy'],
             state['t']]
        return np.array(x, dtype=np.float16)

    def render_mlp(self):

        x_FB3, y_FB3 = self.get_state_FB3()
        xy_initial_FBs, xy_final_FBs, colors_FBs = self.get_states_FBs()

        return x_FB3, y_FB3, xy_initial_FBs, xy_final_FBs, colors_FBs, self.fopt, self.x_offset, self.y_offset, self.path

    def get_states_FBs(self):

        colors = [
            (242, 0, 242),  # Magenta
            (0, 242, 242),  # Cyan
            (242, 242, 0),  # Yellow
            (242, 0, 0),  # Red
            (0, 242, 0),  # Green
            (0, 0, 242),  # Blue
            (242, 121, 0),  # Orange
            (121, 0, 242),  # Purple
            (0, 242, 121),  # Teal
            (121, 121, 121)  # Gray
        ]

        xy_initial = []
        xy_final = []

        for vesseli,colori in zip(self.other_bw_positions.tolist(), colors):
            xi,yi = vesseli
            xy_final.append([xi,yi])

        for vesseli,colori in zip(self.initial_bw_positions.tolist(), colors):
            xi,yi = vesseli
            xy_initial.append([xi,yi])

        return xy_initial,xy_final,colors

    def get_state_FB3(self):
        return self.state['x'], self.state['y']

