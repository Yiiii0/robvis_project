import json
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.spatial.transform
from tqdm import tqdm

T_DIST = 3.3
R_ANG = 90 / 37
ARROW_L = 1
CURVE_PARTS = 1000
TRAJ_DIR = "exploration_trajectory"    # partially re-generated
# All action types: {'RIGHT', 'LEFT', 'BACKWARD', 'FORWARD', 'IDLE', 'CHECKIN', 'QUIT'}
MOVEMENT_ACTS = {"RIGHT", "LEFT", "BACKWARD", "FORWARD"}

def render_traj(pos_hist, heading_hist, map_bounds=(100, 100), visible=None, figsize=(10, 10)):
    traj_fig, traj_ax = plt.subplots(figsize=figsize)
    cumu_pos = np.vstack(pos_hist)
    # Ensure at least a big enough retangle centered at the origin is visible (length and width from map_bounds)
    traj_ax.set_xlim(min(np.min(cumu_pos[:, 0]), -map_bounds[0] - ARROW_L), max(np.max(cumu_pos[:, 0]), map_bounds[0] + ARROW_L))
    traj_ax.set_ylim(min(np.min(cumu_pos[:, 1]), -map_bounds[1] - ARROW_L), max(np.max(cumu_pos[:, 1]), map_bounds[1] + ARROW_L))
    if visible is not None:
        cumu_pos, heading_hist = cumu_pos[visible[0] : visible[1]], heading_hist[visible[0] : visible[1]]
    traj_ax.plot(cumu_pos[:, 0], cumu_pos[:, 1])
    arrow_rot = scipy.spatial.transform.Rotation.from_euler("z", -heading_hist[-1], degrees=True).as_matrix()
    arrow_tip = arrow_rot[:2, 1] * ARROW_L + cumu_pos[-1]
    traj_ax.annotate("", arrow_tip, cumu_pos[-1], arrowprops=dict())
    return traj_fig

def execute_actions(curr_heading, curr_pos, actions, t_dist=T_DIST, r_ang=R_ANG):
    new_pos = curr_pos
    for action in actions:
        if action in ["RIGHT", "LEFT"]:
            curr_heading += r_ang if action == "RIGHT" else -r_ang
        # If need to move, then use heading to move T_DIST in global coordiantes
        elif action in ["FORWARD", "BACKWARD"]:
            rot_mat = scipy.spatial.transform.Rotation.from_euler("z", -curr_heading, degrees=True).as_matrix()
            w_move = rot_mat[:2, 1] * t_dist
            new_pos = new_pos + (w_move if action == "FORWARD" else -w_move)
        elif action == "CHECKIN":
            pass
        elif action == "QUIT":
            pass
    return curr_heading, new_pos

def execute_bulk_actions(actions_fpath):
    # Load actions taken for each timestamp, could be a combination
    with open(actions_fpath, "r") as actions_file:
        actions = json.load(actions_file)
    all_actions = []
    for action_step in actions:
        all_actions.append(action_step["action"])
    
    curr_heading = 0
    curr_pos = np.array([0, 0], dtype=float)
    heading_hist, pos_hist = [curr_heading], [curr_pos]

    step_i = 0
    while step_i < len(all_actions):
        step_actions = all_actions[step_i]
        old_step_i = step_i
        if step_i == 5200:
            curr_heading = -180
        if step_i == 6077:
            curr_heading = 175
        if step_i == 12222:
            step_i = 12304
        if step_i == 12645:
            curr_heading = 90
            step_i = 12701
        if step_i == 13710:
            curr_heading = 0
            step_i = 13750
        if step_i == 13782:
            curr_heading = 90
            step_i = 13826
        if step_i == 14810:
            curr_heading = -90
            step_i = 14853
        if step_i == 15465:
            curr_heading = 180
            step_i = 15485
        if step_i > old_step_i:
            heading_hist.extend([curr_heading for _ in range(step_i - old_step_i)])
            pos_hist.extend([curr_pos for _ in range(step_i - old_step_i)])
        # Treat combined translation and rotation actions as a curve, and break down the curve into small line segments
        if "|" in step_actions[0] and len(set(step_indiv := step_actions[0].split("|")).difference(MOVEMENT_ACTS)) == 0:
            curve_acts = [step_indiv[step_i] for _ in range(CURVE_PARTS) for step_i in range(len(step_indiv))]
            curr_heading, curr_pos = execute_actions(curr_heading, curr_pos, curve_acts, T_DIST / CURVE_PARTS, R_ANG / CURVE_PARTS)
        else:
            curr_heading, curr_pos = execute_actions(curr_heading, curr_pos, step_actions)
        heading_hist.append(curr_heading)
        pos_hist.append(curr_pos)
        step_i += 1
    return pos_hist, heading_hist


if __name__ == "__main__":
    matplotlib.use("agg")

    pos_hist, heading_hist = execute_bulk_actions("exploration_data/data_info.json")
    assert len(pos_hist) == len(heading_hist)
    print(f"In total {len(pos_hist)} moves")

    os.makedirs(TRAJ_DIR, exist_ok=True)
    for step_i in tqdm(range(0, len(pos_hist), 20)):
        traj_fig = render_traj(pos_hist[:step_i + 1], heading_hist[:step_i + 1])
        traj_fig.savefig(os.path.join(TRAJ_DIR, f"{str(step_i).zfill(len(str(len(pos_hist))))}.png"))
        plt.close()
