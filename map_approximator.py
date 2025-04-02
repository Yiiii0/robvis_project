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

    for step_actions in all_actions:
        # Treat combined translation and rotation actions as a curve, and break down the curve into small line segments
        if len(step_actions) > 1 and len(set(step_actions).difference(MOVEMENT_ACTS)) == 0:
            curve_acts = [step_actions[step_i] for _ in range(CURVE_PARTS) for step_i in range(len(step_actions))]
            curr_heading, curr_pos = execute_actions(curr_heading, curr_pos, curve_acts, T_DIST / CURVE_PARTS, R_ANG / CURVE_PARTS)
        else:
            curr_heading, curr_pos = execute_actions(curr_heading, curr_pos, step_actions)
        heading_hist.append(curr_heading)
        pos_hist.append(curr_pos)
    return pos_hist, heading_hist


if __name__ == "__main__":
    matplotlib.use("agg")

    pos_hist, heading_hist = execute_bulk_actions("exploration_data/data_info.json")
    assert len(pos_hist) == len(heading_hist)

    os.makedirs(TRAJ_DIR, exist_ok=True)
    for step_i in tqdm(range(len(pos_hist))):
        traj_fig = render_traj(pos_hist[:step_i + 1], heading_hist[:step_i + 1])
        traj_fig.savefig(os.path.join(TRAJ_DIR, f"{str(step_i).zfill(len(str(len(pos_hist))))}.png"))
        plt.close()
