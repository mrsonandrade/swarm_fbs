# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
from itertools import combinations

L = 38.0
positions_list = [[3.0 * L, 2.5 * (2.0 * L)],  # position 1
                  [3.0 * L, 1.5 * (2.0 * L)],  # position 2
                  [3.0 * L, 0.5 * (2.0 * L)],  # position 3
                  [3.0 * L, -0.5 * (2.0 * L)],  # position 4
                  [3.0 * L, -1.5 * (2.0 * L)],  # position 5
                  [3.0 * L, -2.5 * (2.0 * L)],  # position 6

                  [2.0 * L, 3.0 * (2.0 * L)],  # position 7
                  [2.0 * L, 2.0 * (2.0 * L)],  # position 8
                  [2.0 * L, 1.0 * (2.0 * L)],  # position 9
                  [2.0 * L, 0.0 * (2.0 * L)],  # position 10
                  [2.0 * L, -1.0 * (2.0 * L)],  # position 11
                  [2.0 * L, -2.0 * (2.0 * L)],  # position 12
                  [2.0 * L, -3.0 * (2.0 * L)],  # position 13

                  [1.0 * L, 2.5 * (2.0 * L)],  # position 14
                  [1.0 * L, 1.5 * (2.0 * L)],  # position 15
                  [1.0 * L, 0.5 * (2.0 * L)],  # position 16
                  [1.0 * L, -0.5 * (2.0 * L)],  # position 17
                  [1.0 * L, -1.5 * (2.0 * L)],  # position 18
                  [1.0 * L, -2.5 * (2.0 * L)],  # position 19

                  [0.0 * L, 3.0 * (2.0 * L)],  # position 20
                  [0.0 * L, 2.0 * (2.0 * L)],  # position 21
                  [0.0 * L, 1.0 * (2.0 * L)],  # position 22
                  [0.0 * L, 0.0 * (2.0 * L)],  # position 23
                  [0.0 * L, -1.0 * (2.0 * L)],  # position 24
                  [0.0 * L, -2.0 * (2.0 * L)],  # position 25
                  [0.0 * L, -3.0 * (2.0 * L)]]  # position 26

positions_list = np.asarray(positions_list)

def get_positions(positions, xopt2):
    px = []
    py = []
    for pi in xopt2:
        px.append(positions[:, 0][pi - 1])
        py.append(positions[:, 1][pi - 1])
    return np.asarray([px, py])

alpha = 1.0
beta = 0.5

def anisotropic_distance(p1, p2):
    dx = (p2[0] - p1[0]) * alpha
    dy = (p2[1] - p1[1]) * beta
    return np.sqrt(dx ** 2 + dy ** 2)

def constraint_distances(positions_list, positions, distance_threshold):
    G = nx.Graph()
    for idx in positions:
        G.add_node(idx - 1)

    for i, j in combinations(positions, 2):
        d = anisotropic_distance(positions_list[i - 1], positions_list[j - 1])
        if d <= distance_threshold:
            G.add_edge(i - 1, j - 1)

    components = list(nx.connected_components(G))
    if len(components) == 1 and len(components[0]) == len(positions):
        success = 1
    else:
        success = 0

    return success


def normalize(data):

    if not isinstance(data, np.ndarray): data = np.array(data)

    positions = data[:10]
    period = data[10]
    heading = data[11]

    positions = (np.asarray(positions) - 1.0) / (26.0 - 1.0)
    period = (period - 0.79) / (11.07 - 0.79)
    heading = (heading - 180.0) / (225.0 - 180.0)

    data = [i for i in positions]
    data.append(round(period, 2))
    data.append(round(heading, 2))

    return np.array(data)


def denormalize(data):

    if not isinstance(data, np.ndarray): data = np.array(data)

    positions = data[:10]
    period = data[10]
    heading = data[11]

    positions = positions * (26.0 - 1.0) + 1.0
    period = period * (11.07 - 0.79) + 0.79
    heading = heading * (225.0 - 180.0) + 180.0

    data = [int(i) for i in positions]
    data.append(round(period, 2))
    data.append(round(heading, 2))

    return np.array(data)


def random_X():
    positions = np.random.choice(np.arange(1, 27), size=10, replace=False).tolist()
    period = np.random.uniform(0.79, 11.07)
    heading = np.random.uniform(180.0, 225.0)

    data = positions
    data.append(round(period, 2))
    data.append(round(heading, 2))

    return data


model = joblib.load('surrogate_fbs.pkl')

X_rand = random_X()
X = normalize(X_rand)
column_names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'T_s', 'theta_deg']
X_df = pd.DataFrame(X.reshape(1, -1), columns=column_names)

def objective(x, *args):
    positions = np.round(x).astype(int)
    period, heading = args
    distance_threshold = np.sqrt(L * L + L * L)  # example

    if len(set(positions.tolist())) < 10:
        result = 1e6
    elif 1 == 1:
        x_new = positions.tolist()
        x_new.append(period)
        x_new.append(heading)

        X = normalize(x_new)
        column_names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'T_s', 'theta_deg']
        X_df = pd.DataFrame(X.reshape(1, -1), columns=column_names)

        result = model.predict(X_df)[0]
    else:
        result = 1e6

    return result
