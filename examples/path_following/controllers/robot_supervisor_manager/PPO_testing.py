import numpy as np
import torch
from gym.wrappers import TimeLimit
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from path_following_robot_supervisor import PathFollowingRobotSupervisor


def mask_fn(env):
    return env.get_action_mask()


def run():
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
    deterministic = False  # Whether action is deterministic when testing
    manual_control = False

    # Environment setup
    seed = 1

    maximum_episode_steps = 32_786

    experiment_name = "baseline"
    experiment_description = """Baseline description."""
    experiment_dir = f"./experiments/{experiment_name}"
    load_path = experiment_dir + f"/{experiment_name}_diff_5_agent.zip"

    step_window = 1  # Latest steps of observations
    seconds_window = 0  # How many latest seconds of observations
    add_action_to_obs = True
    reset_on_collisions = 4096  # Allow at least two training steps
    ds_type = "sonar"
    ds_noise = 0.0
    max_ds_range = 100.0  # in cm
    dist_sensors_threshold = 25.0
    on_tar_threshold = 0.1

    tar_d_weight_multiplier = 1.0  # When obstacles are detected, target distance reward is multiplied by this
    tar_a_weight_multiplier = 0.0  # When obstacles are detected, target angle reward is multiplied by this
    tar_dis_weight = 2.0
    tar_ang_weight = 2.0
    ds_weight = 1.0
    obs_turning_weight = 0.0
    tar_reach_weight = 1000.0
    not_reach_weight = 1000.0
    col_weight = 5.0
    time_penalty_weight = 0.1

    # Map setup
    map_w, map_h = 7, 7
    cell_size = None

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = TimeLimit(PathFollowingRobotSupervisor(experiment_description, maximum_episode_steps, step_window=step_window,
                                                 seconds_window=seconds_window,
                                                 add_action_to_obs=add_action_to_obs, max_ds_range=max_ds_range,
                                                 reset_on_collisions=reset_on_collisions, manual_control=manual_control,
                                                 on_target_threshold=on_tar_threshold,
                                                 dist_sensors_threshold=dist_sensors_threshold, ds_type=ds_type,
                                                 ds_noise=ds_noise,
                                                 tar_d_weight_multiplier=tar_d_weight_multiplier,
                                                 tar_a_weight_multiplier=tar_a_weight_multiplier,
                                                 target_distance_weight=tar_dis_weight, tar_angle_weight=tar_ang_weight,
                                                 dist_sensors_weight=ds_weight, obs_turning_weight=obs_turning_weight,
                                                 tar_reach_weight=tar_reach_weight, collision_weight=col_weight,
                                                 time_penalty_weight=time_penalty_weight,
                                                 not_reach_weight=not_reach_weight,
                                                 map_width=map_w, map_height=map_h, cell_size=cell_size, seed=seed),
                    maximum_episode_steps)
    env = ActionMasker(env, action_mask_fn=mask_fn)  # NOQA
    model = MaskablePPO.load(load_path)  # NOQA

    diff_ind = 0
    env.set_difficulty(difficulty_dict[test_difficulty[diff_ind]], test_difficulty[diff_ind])

    obs = env.reset()
    cumulative_rew = 0.0
    tests_count = 0
    tests_per_difficulty = 3
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
                try:
                    print(f"Episode reward: {cumulative_rew}, steps: {steps}, done reason: {info['done_reason']}")
                    done_reasons.append(info['done_reason'])
                except KeyError:
                    print(f"Episode reward: {cumulative_rew}, steps: {steps}, done reason: timeout")
                    done_reasons.append("timeout")
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
