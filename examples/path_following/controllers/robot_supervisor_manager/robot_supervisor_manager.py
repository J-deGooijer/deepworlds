"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_trainer
import PPO_testing

experiment_name = "baseline"
deterministic = False

env = PPO_trainer.run(experiment_name)
PPO_testing.run(experiment_name, env, deterministic)
