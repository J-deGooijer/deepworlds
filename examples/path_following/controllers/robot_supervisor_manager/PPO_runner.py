import os
import numpy as np
import torch
from gym.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from path_following_robot_supervisor import PathFollowingRobotSupervisor


class AdditionalInfoCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, experiment_name, env, current_difficulty=None, verbose=1):
        super(AdditionalInfoCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.experiment_name = experiment_name
        self.current_difficulty = current_difficulty
        self.env = env

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.logger.record("experiment_name", self.experiment_name)
        self.logger.record("difficulty", self.current_difficulty[0])

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        normed_reward = self.env.sum_normed_reward / self.model.n_steps  # NOQA
        self.logger.record("rollout/normalized reward", normed_reward)
        self.env.reset_sum_reward()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def mask_fn(env):
    return env.get_action_mask()


def run():
    # Environment setup
    seed = 1
    total_timesteps = 2_560_000
    n_steps = 5_120  # Number of steps between training, effectively the size of the buffer to train on
    batch_size = 128
    maximum_episode_steps = 25_600
    gamma = 0.995
    target_kl = 0.5
    vf_coef = 0.5
    ent_coef = 0.01
    experiment_name = "baseline"
    experiment_description = """Baseline description."""
    reset_on_collisions = 500
    verbose = False
    manual_control = False
    max_ds_range = 100.0  # in cm
    dist_sensors_threshold = 20
    add_action_to_obs = True
    window_latest_dense = 5  # Latest steps of observations
    window_older_diluted = 10  # How many latest seconds of observations
    on_tar_threshold = 0.1
    tar_dis_weight = 1.0
    tar_ang_weight = 1.0
    ds_weight = 1.0
    tar_reach_weight = 100.0
    col_weight = 2.0
    time_penalty_weight = 1.0
    net_arch = dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])
    # Map setup
    map_w, map_h = 7, 7
    cell_size = None
    difficulty_dict = {"diff_1": {"type": "box", "number_of_obstacles": 4, "min_target_dist": 2, "max_target_dist": 3},
                       "diff_2": {"type": "box", "number_of_obstacles": 6, "min_target_dist": 3, "max_target_dist": 4},
                       "diff_3": {"type": "box", "number_of_obstacles": 8, "min_target_dist": 4, "max_target_dist": 5},
                       "diff_4": {"type": "box", "number_of_obstacles": 10, "min_target_dist": 5, "max_target_dist": 6},
                       "test_diff":
                           {"type": "random", "number_of_obstacles": 25, "min_target_dist": 6, "max_target_dist": 12}}

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = TimeLimit(PathFollowingRobotSupervisor(experiment_description, window_latest_dense=window_latest_dense,
                                                 window_older_diluted=window_older_diluted,
                                                 add_action_to_obs=add_action_to_obs, max_ds_range=max_ds_range,
                                                 reset_on_collisions=reset_on_collisions, manual_control=manual_control,
                                                 verbose=verbose,
                                                 on_target_threshold=on_tar_threshold,
                                                 dist_sensors_threshold=dist_sensors_threshold,
                                                 target_distance_weight=tar_dis_weight, tar_angle_weight=tar_ang_weight,
                                                 dist_sensors_weight=ds_weight, tar_reach_weight=tar_reach_weight,
                                                 collision_weight=col_weight, time_penalty_weight=time_penalty_weight,
                                                 map_width=map_w, map_height=map_h, cell_size=cell_size, seed=seed),
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
    printing_callback = AdditionalInfoCallback(verbose=1, experiment_name=experiment_name, env=env,
                                               current_difficulty=list(difficulty_dict.items())[0])
    env.set_difficulty(difficulty_dict["diff_1"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_1",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_1_agent")
    env.set_difficulty(difficulty_dict["diff_2"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_2",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_2_agent")
    env.set_difficulty(difficulty_dict["diff_3"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_3",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_3_agent")
    env.set_difficulty(difficulty_dict["diff_4"])
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_4",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_4_agent")
    # model = MaskablePPO.load(experiment_dir + f"/{experiment_name}_diff_3_agent")
    env.set_difficulty(difficulty_dict["test_diff"])

    obs = env.reset()
    while True:
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
