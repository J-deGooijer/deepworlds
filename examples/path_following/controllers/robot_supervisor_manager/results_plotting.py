import numpy as np
import matplotlib.pyplot as plt
import json


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


results_path = ""

with open(results_path) as json_file:
    results = json.load(json_file)
    moving_avg_n = len(results["episodes_reward"]) // 10

    # Plot reward per episode
    save_name = None
    reward_per_episode_smoothed = moving_avg(results["episodes_reward"], moving_avg_n)
    multiple_plot([results["episodes_reward"], reward_per_episode_smoothed],
                  ["Reward per episode", "Moving average"],
                  "Reward per episode", "episodes", "reward",
                  save_name)
    # Plot final distance to target per episode
    save_name = None
    episodes_final_distance_smoothed = moving_avg(results["episodes_final_distance"], moving_avg_n)
    multiple_plot([results["episodes_final_distance"], episodes_final_distance_smoothed],
                  ["Final distance per episode", "Moving average"],
                  "Final distance per episode", "episodes", "final distance",
                  save_name)
    # Plot total average action probability per episode
    episodes_avg_action_prob_smoothed = moving_avg(results["episodes_avg_action_prob"], moving_avg_n)
    multiple_plot([results["episodes_avg_action_prob"], episodes_avg_action_prob_smoothed],
                  ["Average action probability per episode", "Moving average"],
                  "Average action probability per episode", "episodes", "average action probability",
                  save_name)

    # Plot each action average probability per episode
    # One line for each action per episode
    save_name = None
    data = [[] for i in range(len(results["episodes_action_probs"][0].keys()))]
    action_names = ["Forward", "Left", "Right", "Stop", "Backwards"]
    for episode in range(len(results["episodes_action_probs"])):
        for action_key in results["episodes_action_probs"][0].keys():
            probs_list = results["episodes_action_probs"][episode][action_key]
            if len(probs_list) == 0:
                data[int(action_key)].append(0.0)
            else:
                data[int(action_key)].append(np.mean(probs_list))
    multiple_plot(data, action_names,
                  "Per action average probability per episode", "episodes", "average action probability")

    save_names = [None for _ in range(action_names)]
    name_ind = 0
    for data_line, action_name in zip(data, action_names):
        data_line_smoothed = moving_avg(data_line, moving_avg_n)
        moving_avg_n = len(data_line) // 10
        multiple_plot([data_line, data_line_smoothed],
                      [f"Average action \"{action_name.lower()}\" probability per episode", "Moving average"],
                      f"Average action \"{action_name.lower()}\" probability per episode",
                      "episodes", f"average action \"{action_name.lower()}\" probability",
                      save_names[name_ind])
        name_ind += 1
