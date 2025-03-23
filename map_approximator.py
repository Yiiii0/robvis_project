import json
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.spatial.transform
from tqdm import tqdm

t_dist = 2    # 50 / 15 from first straight
r_ang = 90 / 37    # 90 / 37 from first corner
ARROW_L = 1
TRAJ_DIR = "exploration_trajectory"

def render_current_traj(pos_hist, curr_heading, map_bounds=(100, 100)):
    traj_fig, traj_ax = plt.subplots(figsize=(10, 10))
    cumu_pos = np.vstack(pos_hist)
    traj_ax.set_xlim(min(np.min(cumu_pos[:, 0]), 0 - ARROW_L), max(np.max(cumu_pos[:, 0]), map_bounds[0] + ARROW_L))
    traj_ax.set_ylim(min(np.min(cumu_pos[:, 1]), 0 - ARROW_L), max(np.max(cumu_pos[:, 1]), map_bounds[1] + ARROW_L))
    traj_ax.plot(cumu_pos[:, 0], cumu_pos[:, 1])
    arrow_rot = scipy.spatial.transform.Rotation.from_euler("z", curr_heading, degrees=True).as_matrix()
    arrow_tip = arrow_rot[:2, 1] * ARROW_L + pos_hist[-1]
    traj_ax.annotate("", arrow_tip, pos_hist[-1], arrowprops=dict(headlength=ARROW_L))
    return traj_fig

def execute_actions(curr_heading, curr_pos, actions):
    new_pos = curr_pos
    for action in actions:
        if action in ["RIGHT", "LEFT"]:
            curr_heading += r_ang if action == "RIGHT" else -r_ang
        elif action in ["FORWARD", "BACKWARD"]:
            rot_mat = scipy.spatial.transform.Rotation.from_euler("z", curr_heading, degrees=True).as_matrix()
            w_move = rot_mat[:2, 1] * t_dist
            new_pos = curr_pos + (w_move if action == "FORWARD" else -w_move)
        elif action == "CHECKIN":
            pass
        elif action == "QUIT":
            pass
    return curr_heading, new_pos


if __name__ == "__main__":
    action_json = "exploration_data/data_info.json"
    with open(action_json, "r") as actions_file:
        actions = json.load(actions_file)

    all_actions = []
    for action_step in actions:
        all_actions.append(action_step["action"])
    # All action types: {'RIGHT', 'LEFT', 'BACKWARD', 'FORWARD', 'IDLE', 'CHECKIN', 'QUIT'}

    plt.ioff()
    os.makedirs(TRAJ_DIR, exist_ok=True)
    curr_heading = 0
    curr_pos = np.array([50, 0], dtype=float)
    heading_hist, pos_hist = [curr_heading], [curr_pos]

    for step_i, step_actions in tqdm(enumerate(all_actions), total=len(all_actions)):
        curr_heading, curr_pos = execute_actions(curr_heading, curr_pos, step_actions)
        heading_hist.append(curr_heading)
        pos_hist.append(curr_pos)
        traj_fig = render_current_traj(pos_hist, curr_heading)
        traj_fig.savefig(os.path.join(TRAJ_DIR, f"{str(step_i).zfill(len(str(len(all_actions))))}.png"))
        plt.close()
