from numpy import convolve, ones, mean
import os
from path_following_robot_supervisor import PathFollowingRobotSupervisor
from agent.PPO_agent import PPOAgent, Transition
from utilities import plot_data


def run():
    # Initialize supervisor object
    env = PathFollowingRobotSupervisor()

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    # We pass the number of inputs and the number of outputs, taken from the gym spaces
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)

    parent_dir = "./experiments"
    experiment_name = "window_1"

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    parent_dir = os.path.join("./experiments", experiment_name)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    episode_count = 0
    episode_limit = 10000
    episodes_per_checkpoint = 250
    solved = False  # Whether the solved requirement is met
    all_episodes_action_probs = []
    all_episodes_avg_action_probs = []
    all_episodes_final_distance = []
    difficulty = {
        0: {"number_of_obstacles": 5, "min_target_dist": 1, "max_target_dist": 1},
        250: {"number_of_obstacles": 5, "min_target_dist": 1, "max_target_dist": 2},
        500: {"number_of_obstacles": 5, "min_target_dist": 1, "max_target_dist": 3},
        750: {"number_of_obstacles": 5, "min_target_dist": 2, "max_target_dist": 3},
        1000: {"number_of_obstacles": 5, "min_target_dist": 2, "max_target_dist": 4},
        1250: {"number_of_obstacles": 10, "min_target_dist": 3, "max_target_dist": 4},
        1500: {"number_of_obstacles": 10, "min_target_dist": 3, "max_target_dist": 5},
        1750: {"number_of_obstacles": 10, "min_target_dist": 4, "max_target_dist": 5},
        2000: {"number_of_obstacles": 10, "min_target_dist": 4, "max_target_dist": 6},
        2250: {"number_of_obstacles": 15, "min_target_dist": 5, "max_target_dist": 6},
        2500: {"number_of_obstacles": 15, "min_target_dist": 5, "max_target_dist": 7},
        2750: {"number_of_obstacles": 15, "min_target_dist": 6, "max_target_dist": 7},
        3000: {"number_of_obstacles": 15, "min_target_dist": 6, "max_target_dist": 8},
        3250: {"number_of_obstacles": 20, "min_target_dist": 7, "max_target_dist": 8},
        3500: {"number_of_obstacles": 20, "min_target_dist": 7, "max_target_dist": 9},
        3750: {"number_of_obstacles": 20, "min_target_dist": 8, "max_target_dist": 9},
        4000: {"number_of_obstacles": 20, "min_target_dist": 8, "max_target_dist": 10},
        4250: {"number_of_obstacles": 25, "min_target_dist": 9, "max_target_dist": 10},
        4500: {"number_of_obstacles": 25, "min_target_dist": 9, "max_target_dist": 11},
        4750: {"number_of_obstacles": 25, "min_target_dist": 10, "max_target_dist": 11},
        5000: {"number_of_obstacles": 25, "min_target_dist": 10, "max_target_dist": 12},
    }
    # Save experiment setup to json file
    env.export_parameters(os.path.join(parent_dir, "experiment_parameters.json"), agent, difficulty, episode_limit)

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episode_limit:
        if episode_count in difficulty.keys():
            env.set_difficulty(difficulty[episode_count])

        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        # This dict holds the probabilities of each chosen action
        episode_action_probs = {str(i): [] for i in range(env.action_space.n)}
        episode_final_distance = 1.0  # This is the final distance reached in the episode normalized in [0.0, 1.0]

        # Inner loop is the episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_action, action_prob = agent.work(state, type_="selectAction")
            # Save the current selected_action's probability
            episode_action_probs[str(selected_action)].append(action_prob)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = env.step(selected_action)

            # Save the current state transition in agent's memory
            trans = Transition(state, selected_action, action_prob, reward, new_state)
            agent.store_transition(trans)

            env.episode_score += reward  # Accumulate episode reward
            if done or step == env.steps_per_episode - 1:
                # Save final distance achieved from final state
                episode_final_distance = state[0]

                agent.train_step(batch_size=step + 1)
                solved = env.solved()  # Check whether the task is solved

                # Save agent
                if episode_count % episodes_per_checkpoint == 0 \
                        and episode_count != 0 \
                        or episode_count == episode_limit - 1:
                    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    agent.save(os.path.join(checkpoint_dir, str(episode_count)))
                break

            state = new_state  # state for next step is current step's new_state
        # End of the episode, print and save some stats
        print(f"{'#':#<50}")
        print(f"Experiment \"{experiment_name}\" - episode #{episode_count + 1}\n ")
        print(f"Total reward: {env.episode_score:.2f}\n ")
        # Save the episode's score
        env.episode_score_list.append(env.episode_score)
        # Save the episode's action probabilities
        all_episodes_action_probs.append(episode_action_probs)
        all_probs = []
        for i in range(env.action_space.n):
            all_probs.extend(episode_action_probs[str(i)])
        # Save the average action probability
        average_episode_action_prob = mean(all_probs)
        all_episodes_avg_action_probs.append(average_episode_action_prob)
        print(f"All actions average probability: {average_episode_action_prob * 100:.2f} %")
        actions = ["forward", "left", "right", "stop", "backward"]
        print("Average probability per action :")
        for i in range(env.action_space.n):
            action = actions[i]
            print(f"{'-':>22}{action:<9}: {mean(episode_action_probs[str(i)]) * 100:.2f} %")
        # Save the episode final distance achieved
        all_episodes_final_distance.append(episode_final_distance)

        episode_count += 1  # Increment episode counter

    try:
        # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
        moving_avg_n = 10
        plot_data(convolve(env.episode_score_list, ones((moving_avg_n,)) / moving_avg_n, mode='valid'),  # NOQA
                  "episode", "episode score", "Episode scores over episodes - " + experiment_name)
    except Exception as e:
        print("Plotting failed:", e)

    from json import dump
    result_dict = {"episodes_reward": env.episode_score_list,
                   "episodes_avg_action_prob": all_episodes_avg_action_probs,
                   "episodes_action_probs": all_episodes_action_probs,
                   "episodes_final_distance": all_episodes_final_distance
                   }
    with open(os.path.join(parent_dir, experiment_name + "_results.json"), 'w') as fp:
        dump(result_dict, fp, indent=4)

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episode_score = 0
    while True:
        for step in range(env.steps_per_episode):
            selected_action, action_prob = agent.work(state, type_="selectActionMax")
            state, reward, done, _ = env.step(selected_action)
            env.episode_score += reward  # Accumulate episode reward

            if done or step == env.steps_per_episode - 1:
                print("Reward accumulated =", env.episode_score)
                env.episode_score = 0
                state = env.reset()
