# Find target & avoid obstacles v2

## General
This is the updated and expanded second version of find and avoid.
For this example, a simple custom differential drive robot is used, equipped with
forward-facing sonar distance sensors. The goal is to navigate to a target by setting
the speeds of the left and right motor, while avoiding obstacles using the distance sensors.
The agent observes its distance to the target as well as its relative facing angle.

## World
The world consists of an arena that can be populated with various obstacles, boxes of various shapes,
chairs and jars. With the help of a grid-map, there are two ways the map can be randomized. The first one
places the robot and the target at a random positions, with variable distance to control difficulty,
and a variable number of obstacles are scattered randomly in the arena. It is made sure that there is 
a free path on the grid map between the robot and the target using a BFS path-finding algorithm.
The second way of randomizing the map, creates a corridor with several rows and three columns. The robot is
placed on the first row and the target is placed on another row which can be controlled to modify difficulty.
Two obstacles are placed in the in-between rows, making sure there is a path between the robot and the target.


## Training
The training takes place in a curriculum-style way, using the corridor randomization, starting with one row of 
obstacles and increasing between training sessions. Finally, the agent is trained in a random map, with all 25 
available obstacles placed within and varying distance between the robot and the target.

## Testing
Testing first takes place in random maps set up similar to the last training session using stable-baselines3's 
evaluation for 100 episodes (and consequently maps). Then a custom evaluation procedure takes place that
tests the agent in various corridor setups and a random map setup, for 100 episodes each.

## Logging
Tensorboard is used to log various aspects of the training procedure. For testing, a CSV file is produced that
includes the reward per episode, how the episode ended (collision, timeout or reached target), 
as well as the steps taken until the end of the episode. Moreover, the sb3 evaluation mean and STD reward 
are recorded.

## Agents 
    
+ [Stable Baselines3 - Contrib Maskable Proximal Policy Optimization (Maskable-PPO)](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
 