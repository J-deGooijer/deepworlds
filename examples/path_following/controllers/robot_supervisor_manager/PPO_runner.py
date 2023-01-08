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

    episode_count = 0
    episode_limit = 1_000_000
    episodes_per_checkpoint = 200
    solved = False  # Whether the solved requirement is met
    avg_episode_action_probs = []  # Save average episode taken actions probability to plot later
    difficulty = {
        0: {"number_of_obstacles": 0, "min_target_dist": 1, "max_target_dist": 1},
        500: {"number_of_obstacles": 4, "min_target_dist": 1, "max_target_dist": 2},
        1500: {"number_of_obstacles": 8, "min_target_dist": 1, "max_target_dist": 3},
        3000: {"number_of_obstacles": 12, "min_target_dist": 1, "max_target_dist": 4},
        5000: {"number_of_obstacles": 16, "min_target_dist": 1, "max_target_dist": 5},
        10000: {"number_of_obstacles": 20, "min_target_dist": 1, "max_target_dist": 6},
        12000: {"number_of_obstacles": 20, "min_target_dist": 2, "max_target_dist": 6},
        14000: {"number_of_obstacles": 20, "min_target_dist": 3, "max_target_dist": 6},
        16000: {"number_of_obstacles": 20, "min_target_dist": 4, "max_target_dist": 6},
        18000: {"number_of_obstacles": 20, "min_target_dist": 5, "max_target_dist": 6},
        20000: {"number_of_obstacles": 20, "min_target_dist": 6, "max_target_dist": 6},
    }

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episode_limit:
        if episode_count in difficulty.keys():
            env.set_difficulty(difficulty[episode_count])

        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        action_probs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_action, action_prob = agent.work(state, type_="selectAction")
            # Save the current selected_action's probability
            action_probs.append(action_prob)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = env.step(selected_action)

            # Save the current state transition in agent's memory
            trans = Transition(state, selected_action, action_prob, reward, new_state)
            agent.store_transition(trans)

            env.episode_score += reward  # Accumulate episode reward
            if done or step == env.steps_per_episode - 1:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                agent.train_step(batch_size=step + 1)
                solved = env.solved()  # Check whether the task is solved

                # Save agent
                if (episode_count + 1) % episodes_per_checkpoint == 0:
                    if not os.path.exists("./checkpoints"):
                        os.mkdir("./checkpoints")
                    agent.save(f"./checkpoints/checkpoint_{episode_count + 1}")
                break

            state = new_state  # state for next step is current step's new_state

        print("Episode #", episode_count + 1, "score:", env.episode_score)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avg_action_prob = mean(action_probs)
        avg_episode_action_probs.append(avg_action_prob)
        print("Avg action prob:", avg_action_prob)

        episode_count += 1  # Increment episode counter

    try:
        # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
        moving_avg_n = 10
        plot_data(convolve(env.episode_score_list, ones((moving_avg_n,)) / moving_avg_n, mode='valid'),  # NOQA
                 "episode", "episode score", "Episode scores over episodes")
        plot_data(convolve(avg_episode_action_probs, ones((moving_avg_n,)) / moving_avg_n, mode='valid'),  # NOQA
                 "episode", "average episode action probability", "Average episode action probability over episodes")
    except Exception as e:
        print("Plotting failed:", e)

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episode_score = 0
    while True:
        selected_action, action_prob = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = env.step(selected_action)
        env.episode_score += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            state = env.reset()
