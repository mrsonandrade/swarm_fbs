# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pso_optimizer import get_positions, positions_list
import time
import torch
import itertools
from vessel import Vessel
from policy import ActorCritic
import os
import glob

initial_positions_index = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
positions_xy = get_positions(positions_list, initial_positions_index)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def run_simulation(num_episodes=1):
    max_steps = 500

    env = Vessel(max_steps=max_steps)

    net = ActorCritic(
        input_dim=env.state_dims,
        output_dim=env.action_dims,
        device=device
    )

    ckpt_pattern = os.path.join('FB_00100000.pt')
    ckpt_list = glob.glob(ckpt_pattern)

    if len(ckpt_list) > 0:
        ckpt_path = ckpt_list[0]
        print(f"Loading checkpoint: {ckpt_path}")

        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False
        )
        net.load_state_dict(checkpoint['model_G_state_dict'])
    else:
        print(f"Warning: No checkpoint found at {ckpt_pattern}")

    net.eval()

    total_reward = 0

    while total_reward<1000.:
        t_FB3, x_FB3, y_FB3 = [], [] ,[]
        xy_initial_FBs, xy_final_FBs = [], []
        colors_FBs = None
        path = None
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            step_counter = 0
            total_reward = 0

            print(f"\n--- Starting Simulation Episode {ep + 1} ---")

            t_FB3i = 0.
            while not done and step_counter < max_steps:
                with torch.no_grad():
                    action, _, _, _ = net.get_action(state, deterministic=True)

                state, reward, done, _ = env.step(action)
                total_reward += reward

                x_FB3i, y_FB3i, xy_initial_FBsi, xy_final_FBsi, colors_FBs, fopt, x_offset, y_offset, pathi = env.render_mlp()
                st.session_state.fopt = fopt
                t_FB3.append(t_FB3i)
                x_FB3.append(x_FB3i)
                y_FB3.append(y_FB3i)
                xy_initial_FBs.append(xy_initial_FBsi)
                xy_final_FBs.append(xy_final_FBsi)

                if path is None:
                    path = np.asarray(pathi)
                    path[:, 0] += x_offset
                    path[:, 1] += y_offset

                step_counter += 1
                t_FB3i += 0.5

    return t_FB3, x_FB3, y_FB3, xy_initial_FBs, xy_final_FBs, colors_FBs, fopt, x_offset, y_offset, path

colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "white"
]

st.set_page_config(page_title="Wave Simulation", layout="wide")
st.markdown(
    """
    <style>
    /* Main content area */
    .block-container {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("A Self-Adaptive Swarm of Floating Breakwaters via Particle Swarm Optimization, RRT, and Reinforcement Learning")
st.sidebar.markdown("Emerson M. de Andrade, Daniel O. Costa, Antonio C. Fernandes, Joel S. Sales Junior")
st.sidebar.markdown("---")  # optional separator
st.sidebar.header("Wave Settings")
theta = st.sidebar.slider("Wave Direction (degrees)", 180, 225, 205, 5, key="theta_slider")
T = st.sidebar.slider("Wave Period (seconds)", 1, 11, 4, 1, key="T_slider")
st.sidebar.markdown("---")  # optional separator
st.sidebar.markdown(
"""
**Notes:**
- The application can be slow since it relies on free computational resources, so optimization may take a few minutes.
- The wave direction controls the propagation angle.
- The period affects the wavelength.
- Changing these values resets the simulation.
- For this site, the PSO iterations were limited to 250, reducing runtime but possibly preventing a globally optimal solution.
"""
)

positions_xy = get_positions(positions_list, [1,2,3,4,5,6,7,8,12,13])
initial_state = None
final_positions = None

if "last_theta" not in st.session_state:
    st.session_state.last_theta = theta
if "fopt" not in st.session_state:
    st.session_state.fopt = None
if "last_T" not in st.session_state:
    st.session_state.last_T = T
if "simulation_started" not in st.session_state:
    st.session_state.simulation_started = False
if "searching_optimal" not in st.session_state:
    st.session_state.searching_optimal = False
if "trajectories_ready" not in st.session_state:
    st.session_state.trajectories_ready = False
if "t" not in st.session_state:
    st.session_state.t = np.linspace(0, 20, 200)
if "trajectories" not in st.session_state:
    st.session_state.trajectories = (
        [xi for xi in positions_xy[0]],
        [yi for yi in positions_xy[1]]
    )

if theta != st.session_state.last_theta or T != st.session_state.last_T:
    st.session_state.simulation_started = False
    st.session_state.searching_optimal = False
    st.session_state.trajectories_ready = False
    st.session_state.t = np.linspace(0, 20, 200)
    st.session_state.trajectories = (
        [xi for xi in positions_xy[0]],
        [yi for yi in positions_xy[1]]
    )
    st.session_state.last_theta = theta
    st.session_state.last_T = T

waterdepth = 50.0
g = 9.81
wave_length = T * np.sqrt(g * waterdepth)
wave_amplitude = 1.0

xlims = [-700, 300]
ylims = [-300, 300]
x = np.linspace(xlims[0], xlims[1], 200)
y = np.linspace(ylims[0], ylims[1], 200)
X, Y = np.meshgrid(x, y)

theta_rad = np.deg2rad(theta)
kx = np.cos(theta_rad) * (2 * np.pi / wave_length)
ky = np.sin(theta_rad) * (2 * np.pi / wave_length)

if st.session_state.searching_optimal:
    button_label = "Preparing simulation..."
    button_color = "background-color: yellow; color: black; cursor: not-allowed;"
else:
    button_label = "Run Simulation"
    button_color = "background-color: green; color: white; cursor: pointer;"


st.markdown(
    f"""
    <style>
    div.stButton > button:first-child {{
        {button_color}
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

if not st.session_state.simulation_started:
    if st.button(button_label):
        st.session_state.simulation_started = True
        st.session_state.searching_optimal = True
        st.rerun()

else:
    st.button(button_label, disabled=True)

    if st.session_state.searching_optimal and not st.session_state.trajectories_ready:

        t_FB3, x_FB3, y_FB3, xy_initial_FBs, xy_final_FBs, colors_FBs, fopt, x_offset, y_offset, path = run_simulation()

        st.session_state.fopt = fopt

        t_FB3 = np.asarray(t_FB3)
        xy_final_FBs = np.array(xy_final_FBs)
        x0 = xy_final_FBs[:, :, 0]
        y0 = xy_final_FBs[:, :, 1]

        st.session_state.t = t_FB3
        x = [x0[:, i]+x_offset for i in range(0, 10)]
        y = [y0[:, i]+y_offset for i in range(0, 10)]

        x_FB3 = np.array(x_FB3)+x_offset
        y_FB3 = np.array(y_FB3)+y_offset
        x.append(x_FB3)
        y.append(y_FB3)

        st.session_state.trajectories = (x, y)

        st.session_state.searching_optimal = False
        st.session_state.trajectories_ready = True

placeholder = st.empty()

frame = 0
while True:

    t_wave = frame * 0.5
    i = min(frame, len(st.session_state.t) - 1)

    #for i in range(len(st.session_state.t)):
    fig, ax = plt.subplots()
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])

    Z = wave_amplitude * np.sin(kx * X + ky * Y - 2 * np.pi * t_wave / T)

    if st.session_state.simulation_started and st.session_state.trajectories_ready:
        x_list, y_list = st.session_state.trajectories
        Z_mu = st.session_state.fopt  # constant value
        ax.set_title(f"t = {t_wave:.2f} s")
        ax.text(-375, 0, f"$K_t$\n{st.session_state.fopt:.2f}", fontsize=10, horizontalalignment="center", verticalalignment="center")
    else:
        x_pos, y_pos = positions_xy
        x_list = [np.full_like(st.session_state.t, xi) for xi in x_pos]
        y_list = [np.full_like(st.session_state.t, yi) for yi in y_pos]
        Z_mu = 1.0

    X_min, X_max = -425., -325.
    Y_min, Y_max = -50., 50.
    mask = (X >= X_min) & (X <= X_max) & (Y >= Y_min) & (Y <= Y_max)
    Z[mask] = Z[mask] * Z_mu

    contour = ax.contourf(X, Y, Z, levels=30, cmap='Blues')
    plt.colorbar(contour, ax=ax, label="Wave Height", shrink=0.65)

    ax.scatter(positions_list[:, 0], positions_list[:, 1], marker='x', s=16, color='black')

    try:
        ax.plot(path[:, 0], path[:, 1], color='red')
        for bw_i in range(11):
            rect_x = x_list[bw_i][i]
            rect_y = y_list[bw_i][i]

            if i<(len(st.session_state.t) - 1):
                if colors[bw_i] == "tab:green":
                    ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                               fill=False, edgecolor=colors[bw_i]))
                elif colors[bw_i] == "white":
                    ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                               fill=True, color="tab:green"))
                else:
                    ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                               fill=True, color=colors[bw_i]))
            if i==(len(st.session_state.t) - 1):
                if colors[bw_i] == "tab:green":
                    ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                               fill=True, color=colors[bw_i]))
                elif colors[bw_i] == "white":
                    pass
                else:
                    ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                               fill=True, color=colors[bw_i]))

            try:
                rect_x = initial_state[bw_i][0]
                rect_y = initial_state[bw_i][1]
                ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                           fill=False, color=colors[bw_i]))
            except:
                pass
    except:
        for bw_i in range(10):
            rect_x = x_list[bw_i][i]
            rect_y = y_list[bw_i][i]
            ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                       fill=True, color=colors[bw_i]))
            try:
                rect_x = initial_state[bw_i][0]
                rect_y = initial_state[bw_i][1]
                ax.add_patch(plt.Rectangle((rect_x - 8, rect_y - 19), 16, 38,
                                           fill=False, color=colors[bw_i]))
            except:
                pass

    ax.add_patch(plt.Rectangle((-375 - 50, 0 - 50), 100, 100, fill=False, color='red'))
    ax.set_aspect('equal')

    placeholder.pyplot(fig)
    plt.close(fig)
    frame += 1
    time.sleep(0.10)
