"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_trainer
import PPO_testing

experiment_name = "baseline"
experiment_description = """Baseline description."""
deterministic = False
manual_control = False
only_test = False  # If true, the trained agent from "experiment_name" will be loaded and evaluated

env = PPO_trainer.run(experiment_name, experiment_description, manual_control=manual_control, only_test=only_test)
PPO_testing.run(experiment_name, env, deterministic)
