import random
from warnings import warn
import numpy as np
from gym.spaces import Box, Discrete
from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range, get_distance_from_target, get_angle_from_target
from controller import Supervisor, Keyboard


class PathFollowingRobotSupervisor(RobotSupervisorEnv):
    """
        *Problem description*
        Target position *tar* is at a distance *d* between *d_min*, *d_max*, somewhere around the robot,
        defined in robot-centric axes.
        The robot should be able to observe its current distance and angle *tar_curr* (*tar_d*, *tar_a*) to target
        position *tar*.
        This tuple *tar_curr* can be taken as ground truth from the simulator at the start and during the episode.
        For the real robot, it can be provided with the ground truth *tar_curr* at the start and then calculate the new
        *obs_t* using its odometry, to update *tar_d*, *tar_a*, after each action taken.
        The goal of the robot is to reach position *tar*, i.e. minimize *tar_d* and *tar_a*, while avoiding obstacles
        along its path, keeping in mind that *tar* might be behind an obstacle. The robot should be able to only move
        forward and turn left and right, i.e. it shouldn't be able to reverse.
        *Simple case*
        In the simple case, the robot observation can be augmented with values from distance
        sensors to enable it to avoid obstacles. Ideally, it should use as few distance sensors as possible, e.g. two
        sensors facing forward, like the ones e-puck has left and right of its camera, to be easily transferable to
        a real robot.
        *Camera case*
        In the camera case, the robots observation can be augmented with image data, preferably the segmentation mask
        of the camera to make it invariant to different environments. In the simulator there is the possibility of
        acquiring ground truth segmentation masks from a camera. On the real robot, an indoor RGB image segmentation
        model can be used to provide input for the agent.
        *Reward*
        The reward should be based on:
        - The distance to the *tar* position, positive reward for minimizing the distance, *r_d*
        - The angle to the *tar* position, positive reward for minimizing the angle, *r_a*
        - Whether the robot collided with an obstacle, negative reward and ?maybe? terminate the episode *r_c*
        *Training*
        - Training can be done in multiple stages, first train the robot to go to a position in an empty arena, with
        just the first two rewards and terminate episode after specific time. If robot reaches the position
        it should stop, then the target can be reset randomly until the episode time is up.
        - Afterwards, add obstacles, e.g. boxes or furniture, and apply the avoiding obstacle reward, terminating
        after hitting on obstacles, with a time limit, same as before.
    """

    def __init__(self, steps_per_episode=10000,
                 on_target_threshold=0.1, on_target_limit=5,
                 dist_sensors_weights=None,
                 target_distance_weight=2.0, tar_angle_weight=4.0,
                 dist_path_weight=1.0, dist_sensors_weight=10.0,
                 tar_stop_weight=1000.0, collision_weight=100.0,
                 map_width=7, map_height=7, cell_size=None):
        """
        TODO docstring
        """
        super().__init__()
        # Set up gym spaces
        self.observation_space = Box(low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                     high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                     dtype=np.float64)
        self.action_space = Discrete(5)

        # Set up various robot components
        self.robot = self.getSelf()

        # Distance sensors are used for the robot to perceive obstacles
        self.distance_sensors = []
        try:
            for ds_name in ["left", "outer_left", "inner_left", "center", "inner_right", "outer_right", "right"]:
                self.distance_sensors.append(self.getDevice("ds_" + ds_name))
                self.distance_sensors[-1].enable(self.timestep)  # NOQA

        except AttributeError:
            warn("\nNo distance sensors initialized.\n ")

        # Touch sensor is used to terminate episode when the robot collides with an obstacle
        self.touch_sensor = self.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)  # NOQA

        # Assuming the robot has at least a distance sensor and all distance sensors have the same max value,
        # this loop grabs the first distance sensor child of the robot and gets the max value it can output
        # from its lookup table. This is used for normalizing the observation.
        self.ds_max = []
        for childNodeIndex in range(self.robot.getField("children").getCount()):
            child = self.robot.getField("children").getMFNode(childNodeIndex)  # NOQA
            if child.getTypeName() == "DistanceSensor":
                self.ds_max.append(child.getField("lookupTable").getMFVec3f(-1)[1])

        # Set up motors
        self.left_motor = self.getDevice("left_wheel")
        self.right_motor = self.getDevice("right_wheel")
        self.set_velocity(0.0, 0.0)

        # Grab target node
        self.target = self.getFromDef("TARGET")

        # Set up misc
        self.steps_per_episode = steps_per_episode
        self.episode_score = 0
        self.episode_score_list = []

        # Target-related stuff
        self.target_position = [0.0, 0.0]
        self.on_target_threshold = on_target_threshold  # Threshold that defines whether robot is considered "on target"
        self.previous_distance = 0.0
        self.previous_angle = 0.0
        self.previous_dist_path = 0.0
        self.previous_angle_path = 0.0
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]

        # The weights can change how critical each sensor is for the final distance sensor reward
        if dist_sensors_weights is None:
            self.dist_sensors_weights = [1.0, 1.25, 1.5, 2.0, 1.5, 1.25, 1.0]
        # Normalize weights to add up to 1
        self.dist_sensors_weights = np.array(self.dist_sensors_weights) / np.sum(self.dist_sensors_weights)

        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": tar_angle_weight,
                                   "dist_path": dist_path_weight, "dist_sensors": dist_sensors_weight,
                                   "tar_stop": tar_stop_weight, "collision": collision_weight}

        self.on_target_counter = 0
        self.on_target_limit = on_target_limit  # The number of steps robot should be on target before the target moves
        self.trigger_done = False  # Used to trigger the done condition
        self.just_reset = True

        # Map
        self.map_width, self.map_height = map_width, map_height
        if cell_size is None:
            self.cell_size = [0.5, 0.5]
        # Center map to (0, 0)
        origin = [-(self.map_width // 2) * self.cell_size[0], (self.map_height // 2) * self.cell_size[1]]
        self.map = Grid(self.map_width, self.map_height, origin, self.cell_size)
        # Find diagonal distance on the map which is the max distance between any two map cells
        dx = self.map.get_world_coordinates(0, 0)[0] - self.map.get_world_coordinates(self.map_width - 1,  # NOQA
                                                                                      self.map_height - 1)[0]  # NOQA
        dy = self.map.get_world_coordinates(0, 0)[1] - self.map.get_world_coordinates(self.map_width - 1,  # NOQA
                                                                                      self.map_height - 1)[1]  # NOQA
        self.max_target_distance = np.sqrt(dx * dx + dy * dy)

        # Obstacle references
        self.all_obstacles = []
        for childNodeIndex in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_obstacles.append(child)

        # Path node references
        self.all_path_nodes = []
        for childNodeIndex in range(self.getFromDef("PATH").getField("children").getCount()):
            child = self.getFromDef("PATH").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_path_nodes.append(child)

        self.number_of_obstacles = 0  # The number of obstacles to use, start with 0
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)

        # Path to target stuff
        self.path_to_target = []
        self.min_target_dist = 1
        self.max_target_dist = 1  # The maximum (manhattan) distance of the target length allowed, starts at 1

        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

    def set_difficulty(self, difficulty_dict):
        self.number_of_obstacles = difficulty_dict["number_of_obstacles"]
        self.min_target_dist = difficulty_dict["min_target_dist"]
        self.max_target_dist = difficulty_dict["max_target_dist"]
        print("Changed difficulty to:", difficulty_dict)

    def get_observations(self):
        """
        This method returns the observation vector of the agent.
        It consists of the distance and angle to the target, as well as the 5 distance sensor values.
        All values are normalized in their respective ranges:
        - Distance is normalized to [0.0, 1.0]
        - Angle is normalized to [-1.0, 1.0]
        - Distance sensor values are normalized to [1.0, 0.0].
          This is done so the input gets a large activation value when the sensor returns
          small values, i.e. an obstacle is close.

        :return: Observation vector
        :rtype: list
        """
        # Target distance
        tar_d = get_distance_from_target(self.robot, self.target)
        tar_d = round(normalize_to_range(tar_d, 0.0, self.max_target_distance, 0.0, 1.0, clip=True), 8)
        # Angle between robot facing and target
        tar_a = get_angle_from_target(self.robot, self.target)
        tar_a = round(normalize_to_range(tar_a, -np.pi, np.pi, -1.0, 1.0, clip=True), 8)
        obs = [tar_d, tar_a]

        # Add distance sensor values
        ds_values = []
        for i in range(len(self.distance_sensors)):
            ds = self.distance_sensors[i]
            ds_values.append(ds.getValue())  # NOQA
            ds_values[-1] = round(normalize_to_range(ds_values[-1], 0, self.ds_max[i], 1.0, 0.0), 8)
        obs.extend(ds_values)
        return obs

    def get_reward(self, action):
        ################################################################################################################
        # Get current values

        # Get distance and angle to target
        current_distance = get_distance_from_target(self.robot, self.target)
        current_angle = get_angle_from_target(self.robot, self.target, is_abs=True)  # NOQA

        # Get minimum distance to path
        current_dist_path, current_path_closest_point = self.find_dist_to_path()
        current_angle_path = get_angle_from_target(self.robot, current_path_closest_point,
                                                   node_mode=False, is_abs=True)

        # Get all distance sensor values
        current_dist_sensors = []  # Values are in range [0, self.ds_max]
        for ds in self.distance_sensors:
            current_dist_sensors.append(ds.getValue())  # NOQA

        ################################################################################################################
        # Reward components

        # Reward for decreasing distance to the target
        # Distance to target reward scales between -1.0 and 1.0, based on how much it increases or decreases
        dist_tar_reward = round(normalize_to_range(self.previous_distance - current_distance,
                                                   -0.0013, 0.0013, -1.0, 1.0), 2)

        # Reward for decreasing angle to the target
        # Angle to target reward is 1.0 for decreasing angle, and -1.0 for increasing angle
        if (self.previous_angle - current_angle) > 0.001:
            ang_tar_reward = 1.0
        elif (self.previous_angle - current_angle) < -0.001:
            ang_tar_reward = -1.0
        else:
            ang_tar_reward = 0.0

        # Bonus reward for reaching the target and for stopping
        tar_stop_reward = 0.0
        if current_distance < self.on_target_threshold:
            if action == 3:
                # If action is stop start counting up to on_target_limit
                if self.on_target_counter >= self.on_target_limit:
                    # Action was stop while on target for on_target_limit steps
                    tar_stop_reward = 1.0
                    self.on_target_counter = 0  # Reset counter
                    self.trigger_done = True  # Terminate episode
                else:
                    self.on_target_counter += 1
            else:
                # Action is not stop, reset counter
                self.on_target_counter = 0
        else:
            # Distance is over threshold, reset counter
            self.on_target_counter = 0

        # Reward for avoiding obstacles
        dist_sensors_rewards = []
        for i in range(len(self.distance_sensors)):
            dist_sensors_rewards.append(normalize_to_range(current_dist_sensors[i], 0.0, self.ds_max[i], -1.0, 0.0)
                                        * self.dist_sensors_weights[i])
        dist_sensors_reward = sum(dist_sensors_rewards)

        # Reward robot for closing the distance to the predefined path
        if (self.previous_dist_path - current_dist_path) > 0.00001:
            dist_path_reward = 0.5
        elif (self.previous_dist_path - current_dist_path) < -0.00001:
            dist_path_reward = -0.5
        else:
            dist_path_reward = 0.0
        # Reward robot for closing the angle to the predefined path
        if (self.previous_angle_path - current_angle_path) > 0.00001:
            ang_path_reward = 0.5
        elif (self.previous_angle_path - current_angle_path) < -0.00001:
            ang_path_reward = -0.5
        else:
            ang_path_reward = 0.0

        dist_path_reward += ang_path_reward

        # Check if the robot has collided with anything, assign negative reward and terminate episode
        if self.touch_sensor.getValue() == 1.0:  # NOQA
            collision_reward = -1.0
            self.trigger_done = True
        else:
            collision_reward = 0.0

        ################################################################################################################
        # Total reward calculation

        weighted_dist_tar_reward = round(self.reward_weight_dict["dist_tar"] * dist_tar_reward, 4)
        weighted_ang_tar_reward = round(self.reward_weight_dict["ang_tar"] * ang_tar_reward, 4)
        weighted_dist_path_reward = round(self.reward_weight_dict["dist_path"] * dist_path_reward, 4)
        weighted_dist_sensors_reward = round(self.reward_weight_dict["dist_sensors"] * dist_sensors_reward, 4)
        weighted_tar_stop_reward = round(self.reward_weight_dict["tar_stop"] * tar_stop_reward, 4)
        weighted_collision_reward = round(self.reward_weight_dict["collision"] * collision_reward, 4)

        # print(f"tar dist : {weighted_dist_tar_reward}")
        # print(f"tar ang  : {weighted_ang_tar_reward}")
        # print(f"tar stop : {weighted_tar_stop_reward}")
        # print(f"path dist: {weighted_dist_path_reward}")
        # print(f"path ang : {ang_path_reward}")
        # print(f"sens dist: {weighted_dist_sensors_reward}")
        # print(f"col obst : {weighted_tar_stop_reward}")

        # Baseline reward is distance and angle to target
        reward = weighted_dist_tar_reward + weighted_ang_tar_reward

        # Add sensors reward
        reward += weighted_dist_sensors_reward

        # Add distance to path reward
        reward += weighted_dist_path_reward

        # Stop reward overrides other rewards if robot is within target threshold
        if current_distance < self.on_target_threshold:
            reward = weighted_tar_stop_reward
        # Collision reward overrides all other rewards, because it means the robot has collided with an obstacle
        if weighted_collision_reward != 0.0:
            reward = weighted_collision_reward
        # print(f"final reward: {reward}")
        # print("-------")
        self.previous_distance = current_distance
        self.previous_angle = current_angle
        self.previous_dist_path = current_dist_path
        self.previous_angle_path = current_angle_path
        self.previous_dist_sensors = current_dist_sensors

        if self.just_reset:
            self.just_reset = False
            return 0.0
        else:
            return reward

    def is_done(self):
        """
        Episode is done when the maximum number of steps per episode is reached, which is handled by the RL training
        loop, or when the robot detects an obstacle at a 0 distance from one of its sensors.
        :return: Whether the episode is done
        :rtype: bool
        """
        if self.trigger_done:
            self.trigger_done = False
            return True
        return False

    def reset(self):
        """
        Resets the simulation using deepbots default reset and re-initializes robot and target positions.
        """
        starting_obs = super().reset()
        # Reset path
        self.path_to_target = None

        # Set robot random rotation
        self.robot.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

        # Randomize obstacles and target
        randomization_successful = False
        while not randomization_successful:
            # Randomize robot and obstacle positions
            self.randomize_map()
            # Set the target in a valid position and find a path to it
            # and repeat until a reachable position has been found for the target
            self.path_to_target = self.get_random_path()
            if self.path_to_target is not None:
                randomization_successful = True
                self.path_to_target = self.path_to_target[1:]  # Remove starting node
        self.place_path(self.path_to_target)
        self.just_reset = True
        return starting_obs

    def solved(self):
        """
        This method checks whether the task is solved, so training terminates.
        Solved condition requires that the average episode score of last 10 episodes is over half the
        theoretical maximum of an episode's reward. Empirical observations show that when this average
        reward per episode is achieved, the agent is well-trained.

        The maximum value is found empirically in various map sizes and is calculated dynamically with
        a linear regression fit based on the current map size assuming the map is square.

        This maximum is infeasible for the agent to achieve as it requires the agent to have a perfect policy
        and that a straight unobstructed path to the target exists from the starting position.
        Thus, it is divided by 2 which in practice proved to be achievable.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        # TODO redo this
        # avg_score_limit = (1317.196 * self.map.size()[0] + 4820.286) * 10
        #
        # if len(self.episode_score_list) >= 10:  # Over 10 episodes thus far
        #     if np.mean(self.episode_score_list[-10:]) > avg_score_limit:  # Last 10 episode scores average value
        #         return True
        return False

    def get_default_observation(self):
        """
        Basic get_default_observation implementation that returns a zero vector
        in the shape of the observation space.
        :return: A list of zeros in shape of the observation space
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def step(self, action):
        action = self.apply_action(action)

        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def apply_action(self, action):
        """
        This method gets an integer action value [0, 1, 2, 3, 4] where each value
        corresponds to an action:
        0: Move forward
        1: Turn left
        2: Turn right
        3: Stop
        4: Move backwards

        :param action: The action to execute
        :type action: int
        :return:
        """
        gas = 0.0
        wheel = 0.0
        key = self.keyboard.getKey()
        if key == ord("W"):
            action = 0
        if key == ord("A"):
            action = 1
        if key == ord("D"):
            action = 2
        if key == ord("S"):
            action = 3
        if key == ord("X"):
            action = 4

        if action == 0:  # Move forward
            gas = 1.0
            wheel = 0.0
        elif action == 1:  # Turn left
            gas = 0.0
            wheel = -1.0
        elif action == 2:  # Turn right
            gas = 0.0
            wheel = 1.0
        elif action == 3:  # Don't move
            gas = 0.0
            wheel = 0.0
        elif action == 4:  # Move backwards
            gas = -1.0
            wheel = 0.0

        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        motor_speeds = [0.0, 0.0]
        motor_speeds[0] = gas + wheel
        motor_speeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        motor_speeds = np.clip(motor_speeds, -4, 4)

        # Apply motor speeds
        self.set_velocity(motor_speeds[0], motor_speeds[1])
        return action

    def set_velocity(self, v_left, v_right):
        """
        Sets the two motor velocities.
        :param v_left: velocity value for left motor
        :type v_left: float
        :param v_right: velocity value for right motor
        :type v_right: float
        """
        self.left_motor.setPosition(float('inf'))  # NOQA
        self.right_motor.setPosition(float('inf'))  # NOQA
        self.left_motor.setVelocity(v_left)  # NOQA
        self.right_motor.setVelocity(v_right)  # NOQA

    def get_distances(self):
        return self.left_distance_sensor.getValue(), self.right_distance_sensor.getValue()  # NOQA

    def randomize_map(self):
        """
        TODO docstring
        """
        self.map.empty()
        self.map.add_random(self.robot)  # Add robot in a random position
        for node in random.sample(self.all_obstacles, self.number_of_obstacles):
            self.map.add_random(node)
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

    def get_random_path(self):
        """
        TODO docstring
        """
        robot_coordinates = self.map.find_by_name("robot")
        if not self.map.add_near(robot_coordinates[0], robot_coordinates[1],
                                 self.target,
                                 min_distance=self.min_target_dist, max_distance=self.max_target_dist):
            return None  # Need to re-randomize obstacles as add_near failed
        return self.map.bfs_path(robot_coordinates, self.map.find_by_name("target"))

    def place_path(self, path):
        for p, l in zip(path, self.all_path_nodes):
            self.map.add_cell(p[0], p[1], l)

    def find_dist_to_path(self):
        def dist_to_line_segm(p, l1, l2):
            v = l2 - l1
            w = p - l1
            c1 = np.dot(w, v)
            if c1 <= 0:
                return np.linalg.norm(p - l1), l1
            c2 = np.dot(v, v)
            if c2 <= c1:
                return np.linalg.norm(p - l2), l2
            b = c1 / c2
            pb = l1 + b * v
            return np.linalg.norm(p - pb), pb

        np_path = np.array([self.map.get_world_coordinates(self.path_to_target[i][0], self.path_to_target[i][1])
                            for i in range(len(self.path_to_target))])
        robot_pos = np.array(self.robot.getPosition()[:2])

        if len(np_path) == 1:
            return np.linalg.norm(np_path[0] - robot_pos), np_path[0]

        min_distance = float('inf')
        closest_point = None
        for i in range(np_path.shape[0] - 1):
            edge = np.array([np_path[i], np_path[i + 1]])
            distance, point_on_line = dist_to_line_segm(robot_pos, edge[0], edge[1])
            min_distance = min(min_distance, distance)
            closest_point = point_on_line
        return min_distance, closest_point

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")


class Grid:
    """
    Partially coded by OpenAI's ChatGPT.
    """

    def __init__(self, width, height, origin, cell_size):
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.origin = origin
        self.cell_size = cell_size

    def size(self):
        return len(self.grid[0]), len(self.grid)

    def add_cell(self, x, y, node):
        if self.grid[y][x] is None and self.is_in_range(x, y):
            self.grid[y][x] = node
            node.getField("translation").setSFVec3f(
                [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], node.getPosition()[2]])
            return True
        return False

    def remove_cell(self, x, y):
        if self.is_in_range(x, y):
            self.grid[y][x] = None
        else:
            warn("Can't remove cell outside grid range.")

    def get_cell(self, x, y):
        if self.is_in_range(x, y):
            return self.grid[y][x]
        else:
            warn("Can't return cell outside grid range.")
            return None

    def get_neighbourhood(self, x, y):
        if self.is_in_range(x, y):
            neighbourhood_coords = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                                    (x + 1, y + 1), (x - 1, y - 1),
                                    (x - 1, y + 1), (x + 1, y - 1)]
            neighbourhood_nodes = []
            for nc in neighbourhood_coords:
                if self.is_in_range(nc[0], nc[1]):
                    neighbourhood_nodes.append(self.get_cell(nc[0], nc[1]))
            return neighbourhood_nodes
        else:
            warn("Can't get neighbourhood of cell outside grid range.")
            return None

    def is_empty(self, x, y):
        if self.is_in_range(x, y):
            if self.grid[y][x]:
                return False
            else:
                return True
        else:
            warn("Coordinates provided are outside grid range.")
            return None

    def empty(self):
        self.grid = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

    def add_random(self, node):
        x = random.randint(0, len(self.grid[0]) - 1)
        y = random.randint(0, len(self.grid) - 1)
        if self.grid[y][x] is None:
            return self.add_cell(x, y, node)
        else:
            self.add_random(node)

    def add_near(self, x, y, node, min_distance=1, max_distance=1):
        # Make sure the randomly selected cell is not occupied
        for tries in range(self.size()[0] * self.size()[1]):
            coords = self.get_random_neighbor(x, y, min_distance, max_distance)
            if coords and self.add_cell(coords[0], coords[1], node):
                return True  # Return success, the node was added
        return False  # Failed to insert near

    def get_random_neighbor(self, x, y, d_min, d_max):
        neighbors = []
        rows = self.size()[0]
        cols = self.size()[1]
        for i in range(-d_max, d_max + 1):
            for j in range(-d_max, d_max + 1):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < rows and 0 <= y + j < cols:
                    distance = abs(x + i - x) + abs(y + j - y)
                    if d_min <= distance <= d_max:
                        neighbors.append((x + i, y + j))
        if len(neighbors) == 0:
            return None
        return random.choice(neighbors)

    def get_world_coordinates(self, x, y):
        if self.is_in_range(x, y):
            world_x = self.origin[0] + x * self.cell_size[0]
            world_y = self.origin[1] - y * self.cell_size[1]
            return world_x, world_y
        else:
            return None, None

    def get_grid_coordinates(self, world_x, world_y):
        x = round((world_x - self.origin[0]) / self.cell_size[0])
        y = -round((world_y - self.origin[1]) / self.cell_size[1])
        if self.is_in_range(x, y):
            return x, y
        else:
            return None, None

    def find_by_name(self, name):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] and self.grid[y][x].getField("name").getSFString() == name:  # NOQA
                    return x, y
        return None

    def is_in_range(self, x, y):
        if (0 <= x < len(self.grid[0])) and (0 <= y < len(self.grid)):
            return True
        return False

    def bfs_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        queue = [(start, [start])]  # (coordinates, path to coordinates)
        visited = set()
        visited.add(start)
        while queue:
            coords, path = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:  # neighbors
                x, y = coords
                x2, y2 = x + dx, y + dy
                if self.is_in_range(x2, y2) and (x2, y2) not in visited:
                    if self.grid[y2][x2] is not None and (x2, y2) == goal:
                        return path + [(x2, y2)]
                    elif self.grid[y2][x2] is None:
                        visited.add((x2, y2))
                        queue.append(((x2, y2), path + [(x2, y2)]))
        return None
