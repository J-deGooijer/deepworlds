from path_following_robot_supervisor import PathFollowingRobotSupervisor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym.wrappers import TimeLimit
import torch
import os


def mask_fn(env):
    return env.get_action_mask()


def run():
    # Environment setup
    total_timesteps = 2_500_000
    n_steps = 5120  # Number of steps between training, effectively the size of the buffer to train on
    batch_size = 128
    maximum_episode_steps = 10240
    gamma = 0.995
    target_kl = None
    vf_coef = 0.5
    ent_coef = 0.01
    experiment_name = "baseline"
    experiment_description = """Window 5-10, sb3"""
    reset_on_collisions = 500
    verbose = False
    manual_control = False
    add_action_to_obs = True
    window_latest_dense = 5  # Latest steps of observations
    window_older_diluted = 10  # How many latest seconds of observations
    on_tar_threshold = 0.1
    tar_dis_weight = 1.0
    tar_ang_weight = 0.5
    ds_weight = 0.5
    tar_reach_weight = 10.0
    col_weight = 2.0
    time_penalty_weight = 0.1
    net_arch = dict(pi=[512, 256, 128], vf=[1024, 512, 256])
    # Map setup
    map_w, map_h = 7, 7
    cell_size = None
    difficulty_dict = {"diff_1": {"type": "box", "number_of_obstacles": 4, "min_target_dist": 2, "max_target_dist": 3},
                       "diff_2": {"type": "box", "number_of_obstacles": 6, "min_target_dist": 3, "max_target_dist": 4},
                       "diff_3": {"type": "box", "number_of_obstacles": 8, "min_target_dist": 4, "max_target_dist": 5},
                       "diff_4": {"type": "box", "number_of_obstacles": 10, "min_target_dist": 5, "max_target_dist": 6},
                       "test_diff":
                           {"type": "random", "number_of_obstacles": 25, "min_target_dist": 6, "max_target_dist": 12}}

    env = TimeLimit(PathFollowingRobotSupervisor(experiment_description, window_latest_dense=window_latest_dense,
                                                 window_older_diluted=window_older_diluted,
                                                 add_action_to_obs=add_action_to_obs,
                                                 reset_on_collisions=reset_on_collisions, manual_control=manual_control,
                                                 verbose=verbose,
                                                 on_target_threshold=on_tar_threshold,
                                                 target_distance_weight=tar_dis_weight, tar_angle_weight=tar_ang_weight,
                                                 dist_sensors_weight=ds_weight, tar_reach_weight=tar_reach_weight,
                                                 collision_weight=col_weight, time_penalty_weight=time_penalty_weight,
                                                 map_width=map_w, map_height=map_h, cell_size=cell_size),
                    maximum_episode_steps)
    env = ActionMasker(env, action_mask_fn=mask_fn)  # NOQA

    experiment_dir = f"./experiments/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    env.export_parameters(experiment_dir + f"/{experiment_name}.json",
                          net_arch, gamma, target_kl, vf_coef, ent_coef,
                          difficulty_dict, maximum_episode_steps)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                        n_steps=n_steps, batch_size=batch_size, gamma=gamma,
                        target_kl=target_kl, vf_coef=vf_coef, ent_coef=ent_coef,
                        verbose=1, tensorboard_log=experiment_dir)
    env.set_difficulty(difficulty_dict["diff_1"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_1")
    model.save(experiment_dir + "/experiment_name_diff_1_agent")
    env.set_difficulty(difficulty_dict["diff_2"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_2")
    model.save(experiment_dir + "/experiment_name_diff_2_agent")
    env.set_difficulty(difficulty_dict["diff_3"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_3")
    model.save(experiment_dir + "/experiment_name_diff_3_agent")
    env.set_difficulty(difficulty_dict["diff_4"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_4")
    model.save(experiment_dir + "/experiment_name_diff_4_agent")
    # model = MaskablePPO.load(experiment_dir + "/experiment_name_diff_4_agent")
    env.set_difficulty(difficulty_dict["test_diff"])

    obs = env.reset()
    while True:
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
