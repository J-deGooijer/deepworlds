import numpy as np
import torch
import random
from sb3_contrib import MaskablePPO


def mask_fn(env):
    return env.get_action_mask()


def run(experiment_name, env, deterministic):
    # Difficulty setup
    difficulty_dict = {"diff_1": {"type": "corridor", "number_of_obstacles": 2,
                                  "min_target_dist": 2, "max_target_dist": 2},
                       "diff_2": {"type": "corridor", "number_of_obstacles": 4,
                                  "min_target_dist": 3, "max_target_dist": 3},
                       "diff_3": {"type": "corridor", "number_of_obstacles": 6,
                                  "min_target_dist": 4, "max_target_dist": 4},
                       "diff_4": {"type": "corridor", "number_of_obstacles": 8,
                                  "min_target_dist": 5, "max_target_dist": 5},
                       "random_diff": {"type": "random", "number_of_obstacles": 25,
                                       "min_target_dist": 5, "max_target_dist": 12}}
    test_difficulty = list(difficulty_dict.keys())

    seed = 2
    env.set_maximum_episode_steps(env.maximum_episode_steps * 2)
    experiment_dir = f"./experiments/{experiment_name}"
    load_path = experiment_dir + f"/{experiment_name}_diff_5_agent.zip"

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = MaskablePPO.load(load_path)  # NOQA

    diff_ind = 0
    env.set_difficulty(difficulty_dict[test_difficulty[diff_ind]], test_difficulty[diff_ind])

    obs = env.reset()
    cumulative_rew = 0.0
    tests_count = 0
    tests_per_difficulty = 100

    print("################### CUSTOM EVALUATION STARTED ###################")
    print(f"Experiment name: {experiment_name}, deterministic: {deterministic}")
    import csv
    header = [experiment_name]
    for i in range(len(test_difficulty)):
        for j in range(tests_per_difficulty):
            header.append(f"{test_difficulty[i]}")

    episode_rewards = ["reward"]
    done_reasons = ["done_reason"]
    steps_row = ["steps"]
    file_name = "/testing_results.csv" if not deterministic else "/testing_results_det.csv"
    with open(experiment_dir + file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        steps = 0
        while True:
            action_masks = mask_fn(env)
            action, _states = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            obs, rewards, done, info = env.step(action)
            steps += 1
            cumulative_rew += rewards
            if done:
                episode_rewards.append(cumulative_rew)
                steps_row.append(steps)
                print(f"Episode reward: {cumulative_rew}, steps: {steps}, done reason: {info['done_reason']}")
                full_test_count = (tests_per_difficulty * diff_ind) + tests_count
                max_test_count = tests_per_difficulty * len(test_difficulty)
                print(f"Testing progress: "
                      f"{round((full_test_count / max_test_count) * 100.0, 2)}"
                      f"%, {full_test_count} / {max_test_count}")
                done_reasons.append(info['done_reason'])
                cumulative_rew = 0.0
                steps = 0

                tests_count += 1
                if tests_count == tests_per_difficulty:
                    diff_ind += 1
                    try:
                        env.set_difficulty(difficulty_dict[test_difficulty[diff_ind]], key=test_difficulty[diff_ind])
                    except IndexError:
                        print("Testing complete.")
                        break
                    tests_count = 0
                env.reset()

        writer.writerow(episode_rewards)
        writer.writerow(done_reasons)
        writer.writerow(steps_row)
    print("################### CUSTOM EVALUATION FINISHED ###################")
