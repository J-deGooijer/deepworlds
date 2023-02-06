# import os
# import pickle
# from numpy import convolve, ones, mean
# from agent.PPO_agent import PPOAgent, Transition
# from utilities import plot_data

from path_following_robot_supervisor import PathFollowingRobotSupervisor
from stable_baselines3 import PPO
from gym.wrappers import TimeLimit
import torch

# Environment setup
total_timesteps = 1_000_000
maximum_episode_steps = 10_000
experiment_description = """Window 10, sb3"""
reset_on_collision = True
verbose = False
action_space_expanded = False
window = 10
on_tar_threshold = 0.1
ds_sensors_weights = None
tar_dis_weight = 1.0
tar_ang_weight = 1.0
path_dis_weight = 0.0
ds_weight = 1.0
tar_reach_weight = 1000.0
col_weight = 1000.0
# Map setup
map_w, map_h = 7, 7
cell_size = None
env = TimeLimit(PathFollowingRobotSupervisor(experiment_description, 0, window, on_tar_threshold,
                                             ds_sensors_weights, tar_dis_weight, tar_ang_weight, path_dis_weight,
                                             ds_weight, tar_reach_weight, col_weight, map_w, map_h, cell_size,
                                             verbose, action_space_expanded, reset_on_collision), maximum_episode_steps)
env.set_difficulty({"number_of_obstacles": 25, "min_target_dist": 5, "max_target_dist": 7})

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[256, 128]))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=total_timesteps)
model.save("mlp_test")

# del model  # remove to demonstrate saving and loading

# model = PPO.load("mlp_test")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        env.reset()
    # env.render()

# def is_solved(reward_list):
#     """
#     This method checks whether the task is solved, so training terminates.
#     Solved condition requires that the average episode score of last 10 episodes is over half the
#     theoretical maximum of an episode's reward. Empirical observations show that when this average
#     reward per episode is achieved, the agent is well-trained.
#
#     The maximum value is found empirically in various map sizes and is calculated dynamically with
#     a linear regression fit based on the current map size assuming the map is square.
#
#     This maximum is infeasible for the agent to achieve as it requires the agent to have a perfect policy
#     and that a straight unobstructed path to the target exists from the starting position.
#     Thus, it is divided by 2 which in practice proved to be achievable.
#
#     :return: True if task is solved, False otherwise
#     :rtype: bool
#     """
#     # TODO redo this
#     # avg_score_limit = (1317.196 * self.map.size()[0] + 4820.286) * 10
#     #
#     # if len(reward_list) >= 10:  # Over 10 episodes thus far
#     #     if np.mean(reward_list[-10:]) > avg_score_limit:  # Last 10 episode scores average value
#     #         return True
#
#
# def print_episode_info(env, experiment_name, episode_count, average_episode_action_prob, episode_metrics):
#     print(f"{'#':#<50}")
#     print(f"Experiment \"{experiment_name}\" - episode #{episode_count + 1}\n ")
#     print("Reward")
#     print(f"{'Total':<12}: {episode_metrics['reward']:.2f}")
#     for key in episode_metrics["reward_sums"].keys():
#         print(f"{key[0].upper() + key[1:]:<12}: {episode_metrics['reward_sums'][key]:.2f}")
#     print(" ")
#     print(f"All actions average probability : {average_episode_action_prob * 100:.2f} %")
#     actions = ["forward", "left", "right", "stop", "backward",
#                "forward-fast", "left-fast", "right-fast", "backwards-fast",
#                "forward-left-fast", "forward-right-fast", "backwards-left-fast", "backwards-right-fast"]
#     for i in range(env.action_space.n):
#         action = actions[i]
#         print(f"Average probability for {action:<22}: {mean(episode_metrics['action_probs'][str(i)]) * 100:.2f} %")
#
#
# def run():
#     load_checkpoint = None  # Set to string to load checkpoint
#     test_diff = 1000  # Difficulty for testing the loaded checkpoint
#     test_action = "selectActionMax"
#
#     save_to_disk = True  # False to disable all saving
#     parent_dir = "./experiments"
#     experiment_name = "window_10_batch_mod_ds"
#     experiment_description = """Window 10, defaults with batch 128 and modified sensors."""
#
#     # Agent setup
#     cuda = True
#     clip_param = 0.02
#     max_grad_norm = 0.05
#     ppo_update_iters = 5
#     gamma = 0.995
#     actor_lr = 3e-4
#     critic_lr = 3e-4
#     # Agent batch size
#     batch_size = 128
#     # When True, train runs only when episode is done, on False it runs when it gets a full batch as set above
#     train_on_done = False
#     # Training setup
#     steps_per_episode = 5000
#     episode_count = 0
#     episode_limit = 10000
#     episodes_per_checkpoint = 250
#     # Environment setup
#     verbose = False
#     action_space_expanded = False
#     window = 10
#     on_tar_threshold = 0.1
#     ds_sensors_weights = None
#     tar_dis_weight = 1.0
#     tar_ang_weight = 2.0
#     path_dis_weight = 0.0
#     ds_weight = 2.0
#     tar_reach_weight = 1000.0
#     col_weight = 1.0
#     # Map setup
#     map_w, map_h = 7, 7
#     cell_size = None
#
#     # Other
#     solved = False  # Whether the solved requirement is met
#     training_metrics = {"total_rewards": [], "rewards_breakdown": [],
#                         "action_probs": [], "avg_action_probs": [],
#                         "final_distances": []
#                         }
#
#     difficulty = {
#         -1: {"number_of_obstacles": 25, "min_target_dist": 1, "max_target_dist": 1},
#         -1: {"number_of_obstacles": 25, "min_target_dist": 1, "max_target_dist": 2},
#         -1: {"number_of_obstacles": 25, "min_target_dist": 1, "max_target_dist": 3},
#         0: {"number_of_obstacles": 25, "min_target_dist": 2, "max_target_dist": 3},
#         75: {"number_of_obstacles": 25, "min_target_dist": 2, "max_target_dist": 4},
#         150: {"number_of_obstacles": 25, "min_target_dist": 3, "max_target_dist": 4},
#         200: {"number_of_obstacles": 25, "min_target_dist": 3, "max_target_dist": 5},
#         300: {"number_of_obstacles": 25, "min_target_dist": 4, "max_target_dist": 5},
#         500: {"number_of_obstacles": 25, "min_target_dist": 4, "max_target_dist": 6},
#         1000: {"number_of_obstacles": 25, "min_target_dist": 5, "max_target_dist": 6},
#         1500: {"number_of_obstacles": 25, "min_target_dist": 5, "max_target_dist": 7},
#         2500: {"number_of_obstacles": 25, "min_target_dist": 6, "max_target_dist": 7},
#         3000: {"number_of_obstacles": 25, "min_target_dist": 6, "max_target_dist": 8},
#         3250: {"number_of_obstacles": 25, "min_target_dist": 7, "max_target_dist": 8},
#         3500: {"number_of_obstacles": 25, "min_target_dist": 7, "max_target_dist": 9},
#         3750: {"number_of_obstacles": 25, "min_target_dist": 8, "max_target_dist": 9},
#         4000: {"number_of_obstacles": 25, "min_target_dist": 8, "max_target_dist": 10},
#         4250: {"number_of_obstacles": 25, "min_target_dist": 9, "max_target_dist": 10},
#         4500: {"number_of_obstacles": 25, "min_target_dist": 9, "max_target_dist": 11},
#         4750: {"number_of_obstacles": 25, "min_target_dist": 10, "max_target_dist": 11},
#         5000: {"number_of_obstacles": 25, "min_target_dist": 10, "max_target_dist": 12},
#     }
#
#     # Initialize supervisor object
#     env = PathFollowingRobotSupervisor(experiment_description, steps_per_episode, window, on_tar_threshold,
#                                        ds_sensors_weights, tar_dis_weight, tar_ang_weight, path_dis_weight,
#                                        ds_weight, tar_reach_weight, col_weight, map_w, map_h, cell_size,
#                                        verbose, action_space_expanded)
#     # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
#     # We pass the number of inputs and the number of outputs, taken from the gym spaces
#     agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, clip_param, max_grad_norm, ppo_update_iters,
#                      batch_size, gamma, cuda, actor_lr, critic_lr)
#
#     if load_checkpoint is not None:
#         agent.load(os.path.join(parent_dir, experiment_name, "checkpoints", str(load_checkpoint)))  # NOQA
#         env.set_difficulty(difficulty[test_diff])
#         solved = True
#
#     # Initialize directories and save experiment parameters
#     if save_to_disk:
#         if not os.path.exists(parent_dir):
#             os.mkdir(parent_dir)
#         parent_dir = os.path.join(parent_dir, experiment_name)
#         if not os.path.exists(parent_dir):
#             os.mkdir(parent_dir)
#         # Save experiment setup to json file
#         env.export_parameters(os.path.join(parent_dir, "experiment_parameters.json"), agent, difficulty, episode_limit)
#
#     # Run outer loop until the episodes limit is reached or the task is solved
#     while not solved and episode_count < episode_limit:
#
#         # Set difficulty
#         if episode_count in difficulty.keys():
#             env.set_difficulty(difficulty[episode_count])
#
#         # Reset robot and get starting observation
#         state = env.reset()
#
#         # Initialize episode_metrics dictionary to save all the metrics
#         episode_metrics = {"reward": 0.0,
#                            "reward_sums": {"target": 0, "sensors": 0, "path": 0, "reach_target": 0, "collision": 0},
#                            "final_distance": 1.0,
#                            "action_probs": {str(i): [] for i in range(env.action_space.n)}
#                            }
#
#         # Episode loop
#         for step in range(env.steps_per_episode):
#             # In training mode the agent samples from the probability distribution, naturally implementing exploration
#             selected_action, action_prob = agent.work(state, type_="selectAction")
#             # Save the current selected_action's probability
#             episode_metrics["action_probs"][str(selected_action)].append(action_prob)
#
#             # Step the supervisor to advance the simulation and get the current selected_action reward,
#             # the new state and whether we reached the done condition
#             new_state, reward, done, _ = env.step(selected_action)
#
#             # Save the current state transition in agent's memory
#             agent.store_transition(Transition(state, selected_action, action_prob, reward["total"], new_state))
#
#             if not train_on_done:
#                 agent.train_step()
#
#             episode_metrics["reward"] += reward["total"]  # Accumulate episode reward
#             for key in episode_metrics["reward_sums"].keys():
#                 episode_metrics["reward_sums"][key] += reward[key]  # Accumulate various rewards
#             if done or step == env.steps_per_episode - 1:
#                 # Save final distance achieved from final state
#                 episode_metrics["final_distance"] = state[0]
#                 if train_on_done:
#                     agent.train_step(batch_size=step + 1)
#                 else:
#                     # When not training on done, train a final time for episode with whatever
#                     # transitions are left in the buffer
#                     if len(agent.buffer) != 0:
#                         agent.train_step(len(agent.buffer))
#                 break
#
#             state = new_state  # state for next step is current step's new_state
#
#         # Check whether the task is solved
#         solved = is_solved([])
#
#         # Save agent and metrics
#         if episode_count % episodes_per_checkpoint == 0 \
#                 and episode_count != 0 \
#                 or episode_count == episode_limit - 1:
#             if save_to_disk:
#                 checkpoint_dir = os.path.join(parent_dir, "checkpoints")
#                 if not os.path.exists(checkpoint_dir):
#                     os.mkdir(checkpoint_dir)
#                 agent.save(os.path.join(checkpoint_dir, str(episode_count)))
#
#                 result_dict = {"episodes_reward": training_metrics["total_rewards"],
#                                "episodes_reward_breakdown": training_metrics["rewards_breakdown"],
#                                "episodes_action_probs": training_metrics["action_probs"],
#                                "episodes_avg_action_prob": training_metrics["avg_action_probs"],
#                                "episodes_final_distance": training_metrics["final_distances"]
#                                }
#                 with open(os.path.join(parent_dir, experiment_name + "_results.pkl"), "wb") as f:
#                     pickle.dump(result_dict, f)
#
#         # End of the episode save metrics
#         # Save the episode's score
#         training_metrics["total_rewards"].append(episode_metrics["reward"])
#         training_metrics["rewards_breakdown"].append(episode_metrics["reward_sums"])
#
#         # Save the episode's action probabilities
#         training_metrics["action_probs"].append(episode_metrics["action_probs"])
#
#         # Calculate and save the average action probability
#         all_probs = []
#         for i in range(env.action_space.n):
#             all_probs.extend(episode_metrics["action_probs"][str(i)])
#         average_episode_action_prob = mean(all_probs)
#         training_metrics["avg_action_probs"].append(average_episode_action_prob)
#
#         # Save the episode final distance achieved
#         training_metrics["final_distances"].append(episode_metrics["final_distance"])
#
#         # Print information
#         print_episode_info(env, experiment_name, episode_count, average_episode_action_prob, episode_metrics)
#
#         episode_count += 1  # Increment episode counter
#
#     # End of training
#     try:
#         # Plot the main convergence metrics of reward per episode and average action probability per episode
#         # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
#         moving_avg_n = 10
#         plot_data(convolve(training_metrics["total_rewards"], ones((moving_avg_n,)) / moving_avg_n,  # NOQA
#                            mode='valid'),
#                   "episode", "episode score", "Episode scores over episodes - " + experiment_name)
#         plot_data(convolve(training_metrics["avg_action_probs"], ones((moving_avg_n,)) / moving_avg_n,  # NOQA
#                            mode='valid'),
#                   "episode", "average action probability", "Average action probability over episodes - "
#                   + experiment_name)
#     except Exception as e:
#         print("Plotting failed:", e)
#
#     result_dict = {"episodes_reward": training_metrics["total_rewards"],
#                    "episodes_reward_breakdown": training_metrics["rewards_breakdown"],
#                    "episodes_action_probs": training_metrics["action_probs"],
#                    "episodes_avg_action_prob": training_metrics["avg_action_probs"],
#                    "episodes_final_distance": training_metrics["final_distances"]
#                    }
#     if save_to_disk:
#         with open(os.path.join(parent_dir, experiment_name + "_results.pkl"), "wb") as f:
#             pickle.dump(result_dict, f)
#
#     if not solved:
#         print("Reached episode limit and task was not solved, deploying agent for testing...")
#     else:
#         print("Task is solved, deploying agent for testing...")
#
#     state = env.reset()
#     while True:
#         episode_metrics = {"reward": 0.0,
#                            "reward_sums": {"target": 0, "sensors": 0, "path": 0, "reach_target": 0, "collision": 0},
#                            "final_distance": 1.0,
#                            "action_probs": {str(i): [] for i in range(env.action_space.n)}
#                            }
#         for step in range(env.steps_per_episode):
#             selected_action, action_prob = agent.work(state, type_=test_action)
#             state, reward, done, _ = env.step(selected_action)
#
#             episode_metrics["action_probs"][str(selected_action)].append(action_prob)
#
#             episode_metrics["reward"] += reward["total"]  # Accumulate episode reward
#             for key in episode_metrics["reward_sums"].keys():
#                 episode_metrics["reward_sums"][key] += reward[key]  # Accumulate various rewards
#
#             if done or step == env.steps_per_episode - 1:
#                 all_probs = []
#                 for i in range(env.action_space.n):
#                     all_probs.extend(episode_metrics["action_probs"][str(i)])
#                 average_episode_action_prob = mean(all_probs)
#                 print_episode_info(env, experiment_name, episode_count, average_episode_action_prob, episode_metrics)
#                 state = env.reset()
#                 break
