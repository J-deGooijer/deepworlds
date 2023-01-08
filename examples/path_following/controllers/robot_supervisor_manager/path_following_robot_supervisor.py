import random
from warnings import warn
import numpy as np
from gym.spaces import Box, Discrete
from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range, get_distance_from_target, get_angle_from_target


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

    def __init__(self):
        """
        TODO docstring
        """
        super().__init__()
        # Set up gym spaces
        self.observation_space = Box(low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                     high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                     dtype=np.float64)
        self.action_space = Discrete(4)

        # Set up various robot components
        self.robot = self.getSelf()

        # try:
        #     self.camera = self.getDevice("camera")
        #     self.camera.enable(self.timestep)  # NOQA
        # except AttributeError:
        #     print("No camera found.")
        self.distance_sensors = []
        try:
            for ds_name in ["outer_left", "inner_left", "center", "inner_right", "outer_right"]:
                self.distance_sensors.append(self.getDevice("ds_" + ds_name))
                self.distance_sensors[-1].enable(self.timestep)  # NOQA

        except AttributeError:
            warn("\nNo distance sensors initialized.\n ")

        self.touch_sensor = self.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)  # NOQA

        # Assuming the robot has at least a distance sensor and all distance sensors have the same max value,
        # this loop grabs the first distance sensor child of the robot and gets the max value it can output
        # from its lookup table.
        self.ds_max = -1
        for childNodeIndex in range(self.robot.getField("children").getCount()):
            child = self.robot.getField("children").getMFNode(childNodeIndex)  # NOQA
            if child.getTypeName() == "DistanceSensor":
                self.ds_max = child.getField("lookupTable").getMFVec3f(-1)[1]
                break

        self.left_motor = self.getDevice("left_wheel")
        self.right_motor = self.getDevice("right_wheel")
        self.set_velocity(0.0, 0.0)

        # Grab target node
        self.target = self.getFromDef("TARGET")

        # Set up misc
        self.steps_per_episode = 5000
        self.episode_score = 0
        self.episode_score_list = []
        self.target_position = [0.0, 0.0]
        self.initial_tar_dist = 0

        self.on_target_threshold = 0.1  # Threshold that defines whether robot is considered "on target"
        self.facing_target_threshold = np.pi / 16  # Threshold on which robot is considered facing the target, π/32~5deg
        self.previous_distance = 0.0
        self.previous_angle = 0.0
        self.on_target_counter = 0
        self.on_target_limit = 400  # The number of steps robot should be on target before the target moves
        self.trigger_done = False

        # Map
        width, height = 7, 7
        cell_size = [0.5, 0.5]
        # Center map to (0, 0)
        origin = [-(width // 2) * cell_size[0], (height // 2) * cell_size[1]]
        self.map = Grid(width, height, origin, cell_size)

        # Obstacle references
        self.all_obstacles = []
        for childNodeIndex in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_obstacles.append(child)

        self.number_of_obstacles = len(self.all_obstacles)  # The number of obstacles to use
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)

        self.path_to_target = None
        self.min_path_length = 2
        self.max_path_length = 5  # The maximum path length allowed

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
        tar_d = round(normalize_to_range(tar_d, 0.0, self.initial_tar_dist, 0.0, 1.0, clip=True), 2)
        # Angle between robot facing and target
        tar_a = get_angle_from_target(self.robot, self.target)
        tar_a = round(normalize_to_range(tar_a, -np.pi, np.pi, -1.0, 1.0, clip=True), 2)
        obs = [tar_d, tar_a]

        # Add distance sensor values
        ds_values = []
        for ds in self.distance_sensors:
            ds_values.append(ds.getValue())  # NOQA
            ds_values[-1] = round(normalize_to_range(ds_values[-1], 0, self.ds_max, 1.0, 0.0, clip=True), 2)
            print(ds_values)
        obs.extend(ds_values)
        return obs

    def get_reward(self, action):
        """
        TODO docstring
        :param action:
        :return:
        """
        r = 0
        current_distance = get_distance_from_target(self.robot, self.target)
        current_angle = get_angle_from_target(self.robot, self.target, is_abs=True)

        if current_distance < self.on_target_threshold and current_angle < self.facing_target_threshold:
            # When on target and facing it, action should be "no action"
            if action != 3:
                r = -10
            else:
                r = 10
        elif current_distance < self.on_target_threshold:
            # Close to target but not facing it, no reward regardless of the action
            # If the robot turns towards the target the decreasing angle reward section below will reward it
            r = 0
        else:
            # Robot is far from the target
            # Distance is decreasing and robot is moving forward
            if current_distance - self.previous_distance < -0.0001 and action == 0:
                # Cumulative reward based on the facing angle
                if current_angle < self.facing_target_threshold:
                    r = r + 3
            # Distance is increasing and robot is moving forward
            elif current_distance - self.previous_distance > 0.0001 and action == 0:
                r = r - 10
            # Distance is neither increasing nor decreasing
            else:
                r = r - 1
        self.previous_distance = current_distance

        # Decreasing angle to target reward
        if current_angle - self.previous_angle < -0.001:
            r = r + 2
        elif current_angle - self.previous_angle > 0.001:
            r = r - 4
        self.previous_angle = current_angle

        # The following section checks whether the robot is on target and facing it
        # for a length of time.
        # If robot is on target and facing it
        if get_distance_from_target(self.robot, self.target) < self.on_target_threshold and \
                get_angle_from_target(self.robot, self.target, is_abs=True) < self.facing_target_threshold:
            # Count to limit
            if self.on_target_counter >= self.on_target_limit:
                # Robot is on target for a number of steps so reward it and reset target to a new position
                self.trigger_done = True
                r += 1000
                self.on_target_counter = 0
            else:
                self.on_target_counter += 1
        # If either distance or angle becomes larger than thresholds, reset counter
        if get_distance_from_target(self.robot, self.target) > self.on_target_threshold or \
                get_angle_from_target(self.robot, self.target, is_abs=True) > self.facing_target_threshold:
            self.on_target_counter = 0

        # Check if the robot has collided with anything
        if self.touch_sensor.getValue() == 1.0:  # NOQA
            self.trigger_done = True
            r = -1000

        return r

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
        self.initial_tar_dist = get_distance_from_target(self.robot, self.target)

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

        return starting_obs

    def solved(self):
        """
        This method checks whether the task is solved, so training terminates.
        Solved condition requires that the average episode score of last 10 episodes is over half the
        theoretical maximum of an episode's reward. Empirical observations show that when this average
        reward per episode is achieved, the agent is well-trained.

        The theoretical maximum is calculated as:
        1. The steps_per_episode divided by the number of steps before the target is considered found (on_target_limit).
           This gives the theoretical number of times the target can be found in an episode.
        2. The previous value multiplied by a 1000 plus the on_target_limit steps multiplied by 10, which is the reward
           per step that the agent is on target and stopped.

        This maximum is infeasible for the agent to achieve as it requires the agent to start on the target already
        every time the target is randomly moved, and thus it is divided by 2 which in practice proved to be achievable.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        avg_score_limit = (self.steps_per_episode // self.on_target_limit) * \
                          (1000 + self.on_target_limit * 10) \
                          / 2

        if len(self.episode_score_list) >= 10:  # Over 10 episodes thus far
            if np.mean(self.episode_score_list[-10:]) > avg_score_limit:  # Last 10 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Basic get_default_observation implementation that returns a zero vector
        in the shape of the observation space.
        :return: A list of zeros in shape of the observation space
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def apply_action(self, action):
        """
        This method gets an integer action value [0, 1, 2, 3] where each value
        corresponds to an action:
        0: Move forward
        1: Turn left
        2: Turn right
        3: Stop

        :param action: The action to execute
        :type action: int
        :return:
        """
        if action == 0:  # Move forward
            gas = 1.0
            wheel = 0.0
        elif action == 1:  # Turn left
            gas = 0.0
            wheel = -1.0
        elif action == 2:  # Turn right
            gas = 0.0
            wheel = 1.0
        else:  # Don't move
            gas = 0.0
            wheel = 0.0

        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        motor_speeds = [0.0, 0.0]
        motor_speeds[0] = gas + wheel
        motor_speeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        motor_speeds = np.clip(motor_speeds, -4, 4)

        # Apply motor speeds
        self.set_velocity(motor_speeds[0], motor_speeds[1])

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
        self.map.insert_random(self.robot)  # Add robot in a random position
        for node in random.sample(self.all_obstacles, self.number_of_obstacles):
            self.map.insert_random(node)
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

    def get_random_path(self):
        """
        TODO docstring
        """
        robot_coordinates = self.map.find_by_name("robot")
        try:
            if not self.map.insert_near(robot_coordinates[0], robot_coordinates[1],
                                        self.target,
                                        min_distance=self.min_path_length, max_distance=self.max_path_length):
                return None  # Need to re-randomize obstacles as insert_near failed
        except RecursionError:
            print("insert_near reached recursion limit error.")
            return None
        return self.map.bfs_path(robot_coordinates, self.map.find_by_name("target"))

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

    def set_cell(self, x, y, node):
        if self.grid[y][x] is None:
            self.grid[y][x] = node
            node.getField("translation").setSFVec3f(
                [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], node.getPosition()[2]])
            return True
        return False

    def remove_cell(self, x, y):
        self.grid[y][x] = None

    def empty(self):
        self.grid = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

    def insert_random(self, node):
        x = random.randint(0, len(self.grid[0]) - 1)
        y = random.randint(0, len(self.grid) - 1)
        if self.grid[y][x] is None:
            return self.set_cell(x, y, node)
        else:
            self.insert_random(node)

    def insert_near(self, x, y, node, min_distance=1, max_distance=1):
        how_near = random.randrange(min_distance, max_distance + 1)
        near_coords = [(x + how_near, y), (x - how_near, y), (x, y + how_near), (x, y - how_near),
                       (x + how_near, y + how_near), (x - how_near, y - how_near),
                       (x - how_near, y + how_near), (x + how_near, y - how_near)]
        coords = random.choice(near_coords)
        # Make sure the random selection is in range
        # Re-randomize everything to make it faster in edge cases
        while not self.is_in_range(coords[0], coords[1]):
            how_near = random.randrange(min_distance, max_distance + 1)
            near_coords = [(x + how_near, y), (x - how_near, y), (x, y + how_near), (x, y - how_near),
                           (x + how_near, y + how_near), (x - how_near, y - how_near),
                           (x - how_near, y + how_near), (x + how_near, y - how_near)]
            coords = random.choice(near_coords)

        if self.set_cell(coords[0], coords[1], node):
            pass
        else:
            self.insert_near(x, y, node)
        return True

    def get_world_coordinates(self, x, y):
        world_x = self.origin[0] + x * self.cell_size[0]
        world_y = self.origin[1] - y * self.cell_size[1]
        return world_x, world_y

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
        queue = [(start, [start])]  # (coordinates, path)
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
