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
        self.logger.record("difficulty", self.current_difficulty)

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
        self.logger.record("general/experiment_name", self.experiment_name)
        self.logger.record("general/difficulty", self.current_difficulty)
        self.logger.record("resets/reset count", self.env.reset_count)
        self.logger.record("resets/reach target count", self.env.reach_target_count)
        self.logger.record("resets/collision termination count", self.env.collision_termination_count)
        self.logger.record("resets/timeout count", self.env.timeout_count)
        if self.env.reach_target_count == 0 or self.env.reset_count == 0:
            self.logger.record("resets/success percentage", 0.0)
        else:
            self.logger.record("resets/success percentage", self.env.reach_target_count / self.env.reset_count)

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
    manual_control = False

    # Environment setup
    seed = 1

    n_steps = 32_786  # Number of steps between training, effectively the size of the buffer to train on
    batch_size = 2048
    maximum_episode_steps = 8_192  # Minimum 4 (8192*4=32768) full episodes per training step
    total_timesteps = 524_288  # Minimum 64 (8192*64=524288) episodes' worth of timesteps per difficulty

    gamma = 0.99
    gae_lambda = 0.95
    target_kl = None
    vf_coef = 0.5
    ent_coef = 0.001

    experiment_name = "Baseline"
    experiment_description = """Baseline description."""
    experiment_dir = f"./experiments/{experiment_name}"

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

    net_arch = dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])
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

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    env.export_parameters(experiment_dir + f"/{experiment_name}.json",
                          net_arch, gamma, gae_lambda, target_kl, vf_coef, ent_coef,
                          difficulty_dict, maximum_episode_steps, n_steps, batch_size)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                        n_steps=n_steps, batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda,
                        target_kl=target_kl, vf_coef=vf_coef, ent_coef=ent_coef,
                        verbose=1, tensorboard_log=experiment_dir)

    printing_callback = AdditionalInfoCallback(verbose=1, experiment_name=experiment_name, env=env,
                                               current_difficulty="diff_1")
    # Corridor 1 training session
    env.set_difficulty(difficulty_dict["diff_1"])
    printing_callback.current_difficulty = "diff_1"
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_1",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_1_agent")
    # Corridor 2 training session
    env.set_difficulty(difficulty_dict["diff_2"])
    printing_callback.current_difficulty = "diff_2"
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_2",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_2_agent")
    # Corridor 3 training session
    env.set_difficulty(difficulty_dict["diff_3"])
    printing_callback.current_difficulty = "diff_3"
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_3",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_3_agent")
    # Corridor 4 training session
    env.set_difficulty(difficulty_dict["diff_4"])
    printing_callback.current_difficulty = "diff_4"
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_4",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_4_agent")
    # Random map training session
    env._max_episode_steps *= 2  # NOQA
    env.maximum_episode_steps = env._max_episode_steps  # NOQA
    env.set_difficulty(difficulty_dict["random_diff"])
    printing_callback.current_difficulty = "random_diff"
    model.learn(total_timesteps=total_timesteps, tb_log_name="difficulty_5",
                reset_num_timesteps=False, callback=printing_callback)
    model.save(experiment_dir + f"/{experiment_name}_diff_5_agent")
    print("Training finished.")
