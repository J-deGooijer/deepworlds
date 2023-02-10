from path_following_robot_supervisor import PathFollowingRobotSupervisor
from stable_baselines3 import PPO
from gym.wrappers import TimeLimit
import torch


def run():
    # Environment setup
    total_timesteps = 500_000
    maximum_episode_steps = 25_000
    experiment_name = "simple_ppo_test_defaults"
    experiment_description = """Window 10, sb3"""
    reset_on_collision = True
    verbose = False
    manual_control = False
    window = 10
    on_tar_threshold = 0.1
    tar_dis_weight = 1.0
    tar_ang_weight = 1.0
    ds_weight = 1.0
    tar_reach_weight = 100.0
    col_weight = 100.0
    # Map setup
    map_w, map_h = 7, 7
    cell_size = None
    env = TimeLimit(PathFollowingRobotSupervisor(experiment_description, obs_window_size=window,
                                                 reset_on_collision=reset_on_collision, manual_control=manual_control,
                                                 verbose=verbose,
                                                 on_target_threshold=on_tar_threshold,
                                                 target_distance_weight=tar_dis_weight, tar_angle_weight=tar_ang_weight,
                                                 dist_sensors_weight=ds_weight, tar_reach_weight=tar_reach_weight,
                                                 collision_weight=col_weight,
                                                 map_width=map_w, map_height=map_h, cell_size=cell_size), maximum_episode_steps)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[256, 128, 64], vf=[512, 256, 128]))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=0.995, verbose=1,
                tensorboard_log="./experiments/" + experiment_name)
    # env.set_difficulty({"type": "box", "number_of_obstacles": 4, "min_target_dist": 2, "max_target_dist": 3})
    # model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_1")
    # env.set_difficulty({"type": "box", "number_of_obstacles": 6, "min_target_dist": 3, "max_target_dist": 4})
    # model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_2")
    # env.set_difficulty({"type": "box", "number_of_obstacles": 8, "min_target_dist": 4, "max_target_dist": 5})
    # model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_3")
    env.set_difficulty({"type": "box", "number_of_obstacles": 10, "min_target_dist": 5, "max_target_dist": 6})
    # model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_4")
    # model.save("mlp_test_sparse")
    model = PPO.load("mlp_test_sparse")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
