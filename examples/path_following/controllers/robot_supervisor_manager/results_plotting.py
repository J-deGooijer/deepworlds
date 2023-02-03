import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from os.path import join


def multiple_plot(data_, labels=None,
                  title="Title", x_label="x_label", y_label="y_label",
                  save_path=None):
    if labels is None:
        labels = [f"label{i}" for i in range(len(data_))]
    x_axis = [i for i in range(len(data_[0]))]
    for data_line_index in range(len(data_)):
        plt.plot(x_axis, data_[data_line_index], label=labels[data_line_index])

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def moving_avg(data_, n, mode="valid"):
    data_padded = np.pad(data_, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(data_padded, np.ones((n,)) / n, mode=mode)


# Change the following paths if needed
parent_dir = "./experiments"
experiment_name = "window_1"
experiment_folder = join(parent_dir, experiment_name)
results_path = join(experiment_folder, experiment_name + "_results.pkl")

read_flag = "r" if results_path.split('.')[-1] == "json" else "rb"
with open(results_path, read_flag) as f:
    if read_flag == "r":
        results = json.load(f)
    else:
        results = pickle.load(f)
    moving_avg_n = len(results["episodes_reward"]) // 10

    # Plot reward per episode
    save_name = join(experiment_folder, "reward.png")
    reward_per_episode_smoothed = moving_avg(results["episodes_reward"], moving_avg_n)
    multiple_plot([results["episodes_reward"], reward_per_episode_smoothed],
                  ["Reward per episode", "Moving average"],
                  "Reward per episode", "episodes", "reward",
                  save_name)
    # TODO Plot reward breakdown per episode
    # Plot final distance to target per episode
    save_name = join(experiment_folder, "dist.png")
    episodes_final_distance_smoothed = moving_avg(results["episodes_final_distance"], moving_avg_n)
    multiple_plot([results["episodes_final_distance"], episodes_final_distance_smoothed],
                  ["Final distance per episode", "Moving average"],
                  "Final distance per episode", "episodes", "final distance",
                  save_name)
    # Plot total average action probability per episode
    save_name = join(experiment_folder, "avg_act.png")
    episodes_avg_action_prob_smoothed = moving_avg(results["episodes_avg_action_prob"], moving_avg_n)
    multiple_plot([results["episodes_avg_action_prob"], episodes_avg_action_prob_smoothed],
                  ["Average action probability per episode", "Moving average"],
                  "Average action probability per episode", "episodes", "average action probability",
                  save_name)

    # Plot each action average probability per episode
    # One line for each action per episode
    data = [[] for i in range(len(results["episodes_action_probs"][0].keys()))]
    actions = ["forward", "left", "right", "stop", "backward",
               "forward-fast", "left-fast", "right-fast", "backwards-fast",
               "forward-left-fast", "forward-right-fast", "backwards-left-fast", "backwards-right-fast"]
    for episode in range(len(results["episodes_action_probs"])):
        for action_key in results["episodes_action_probs"][0].keys():
            probs_list = results["episodes_action_probs"][episode][action_key]
            if len(probs_list) == 0:
                data[int(action_key)].append(0.0)
            else:
                data[int(action_key)].append(np.mean(probs_list))
    multiple_plot(data, actions,
                  "Per action average probability per episode", "episodes", "average action probability")

    save_names = [join(experiment_folder, f"{name.lower()}.png") for name in actions]
    name_ind = 0
    for data_line, action_name in zip(data, actions):
        data_line_smoothed = moving_avg(data_line, moving_avg_n)
        moving_avg_n = len(data_line) // 10
        multiple_plot([data_line, data_line_smoothed],
                      [f"Average action \"{action_name.lower()}\" probability per episode", "Moving average"],
                      f"Average action \"{action_name.lower()}\" probability per episode",
                      "episodes", f"average action \"{action_name.lower()}\" probability",
                      save_names[name_ind])
        name_ind += 1
