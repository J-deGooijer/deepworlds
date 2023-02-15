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

    def __init__(self, description, window_latest_dense=1, window_older_diluted=1, add_action_to_obs=True,
                 max_ds_range=150.0, reset_on_collisions=0, manual_control=False, verbose=False,
                 on_target_threshold=0.1, dist_sensors_threshold=0.0, ds_type="generic",
                 tar_d_weight_multiplier=1.0, tar_a_weight_multiplier=1.0,
                 target_distance_weight=1.0, tar_angle_weight=1.0, dist_sensors_weight=1.0,
                 tar_reach_weight=1.0, collision_weight=1.0, time_penalty_weight=1.0,
                 map_width=7, map_height=7, cell_size=None, seed=None):
        """
        TODO docstring
        """
        super().__init__()
        if seed is not None:
            random.seed(seed)
        self.experiment_desc = description
        self.verbose = verbose
        self.manual_control = manual_control

        # Viewpoint stuff used to reset camera position
        self.viewpoint = self.getFromDef("VIEWPOINT")
        self.viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()

        # Keyboard control
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # Set up various robot components
        self.robot = self.getSelf()
        self.number_of_distance_sensors = 13  # Fixed according to ds that exist on robot

        # Set up gym observation and action spaces
        self.action_names = ["Forward", "Left", "Right", "Stop", "Backward"]
        self.action_space = Discrete(4)  # Actions: Forward, Left, Right, Backward

        self.add_action_to_obs = add_action_to_obs
        self.window_latest_dense = window_latest_dense
        self.window_older_diluted = window_older_diluted
        self.obs_list = []
        # Distance to target, angle to target, distance change, angle change
        single_obs_low = [0.0, -1.0, -1.0, -1.0]
        # Add action "one-hot"
        if self.add_action_to_obs:
            single_obs_low.extend([0.0 for _ in range(self.action_space.n)])
        # Append distance sensor values
        single_obs_low.extend([0.0 for _ in range(self.number_of_distance_sensors)])

        single_obs_high = [1.0, 1.0, 1.0, 1.0]
        if self.add_action_to_obs:
            single_obs_high.extend([1.0 for _ in range(self.action_space.n)])
        single_obs_high.extend([1.0 for _ in range(self.number_of_distance_sensors)])

        self.single_obs_size = len(single_obs_low)
        obs_low = []
        obs_high = []
        for _ in range(self.window_latest_dense + self.window_older_diluted):
            obs_low.extend(single_obs_low)
            obs_high.extend(single_obs_high)
            self.obs_list.extend([0.0 for _ in range(self.single_obs_size)])
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.window_older_diluted * int(np.ceil(1000 / self.timestep))) +
                                          self.window_latest_dense)]

        self.observation_counter_limit = int(np.ceil(1000 / self.timestep))
        self.observation_counter = self.observation_counter_limit
        self.observation_space = Box(low=np.array(obs_low),
                                     high=np.array(obs_high),
                                     dtype=np.float64)

        # Dictionary with distance sensor values as key and masked action as value
        self.action_masks = {}

        # Set up sensors
        self.distance_sensors = []
        self.ds_max = []
        self.dist_sensors_threshold = dist_sensors_threshold
        # Loop through the ds_group node to get max sensor values and initialize the devices
        robot_children = self.robot.getField("children")
        for childNodeIndex in range(robot_children.getCount()):
            robot_child = robot_children.getMFNode(childNodeIndex)  # NOQA
            if robot_child.getTypeName() == "Group":
                ds_group = robot_child.getField("children")
                for i in range(self.number_of_distance_sensors):
                    self.distance_sensors.append(self.getDevice(f"distance sensor({str(i)})"))
                    self.distance_sensors[-1].enable(self.timestep)  # NOQA
                    ds_node = ds_group.getMFNode(i)
                    ds_node.getField("lookupTable").setMFVec3f(-1, [max_ds_range / 100.0, max_ds_range, 0.0])
                    ds_node.getField("type").setSFString(ds_type)
                    self.ds_max.append(max_ds_range)  # NOQA

        # Touch sensor is used to determine when the robot collides with an obstacle
        self.touch_sensor = self.getDevice("touch sensor")
        self.touch_sensor.enable(self.timestep)  # NOQA

        # Set up motors
        self.left_motor = self.getDevice("left_wheel")
        self.right_motor = self.getDevice("right_wheel")
        self.motor_speeds = [0.0, 0.0]
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])

        # Grab target node
        self.target = self.getFromDef("TARGET")

        # Set up misc
        self.on_target_threshold = on_target_threshold  # Threshold that defines whether robot is considered "on target"
        # Various metrics, for the current step and the previous step
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_action = None

        # Dictionary holding the weights for the various reward components
        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": tar_angle_weight,
                                   "dist_sensors": dist_sensors_weight, "tar_reach": tar_reach_weight,
                                   "collision": collision_weight, "time_penalty_weight": time_penalty_weight}
        self.tar_d_weight_multiplier = tar_d_weight_multiplier
        self.tar_a_weight_multiplier = tar_a_weight_multiplier
        self.sum_normed_reward = 0.0  # Used as a metric
        self.collisions_counter = 0
        self.reset_on_collisions = reset_on_collisions  # Whether to reset on collision
        self.trigger_done = False  # Used to trigger the done condition
        self.just_reset = True  # Whether the episode was just reset

        # Map stuff
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

        # Obstacle references and starting positions used to reset them
        self.all_obstacles = []
        self.all_obstacles_starting_positions = []
        for childNodeIndex in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_obstacles.append(child)
            self.all_obstacles_starting_positions.append(child.getField("translation").getSFVec3f())

        # Wall references
        self.walls = [self.getFromDef("WALL_1"), self.getFromDef("WALL_2")]
        self.walls_starting_positions = [self.getFromDef("WALL_1").getField("translation").getSFVec3f(),
                                         self.getFromDef("WALL_2").getField("translation").getSFVec3f()]

        # Path node references and starting positions used to reset them
        self.all_path_nodes = []
        self.all_path_nodes_starting_positions = []
        for childNodeIndex in range(self.getFromDef("PATH").getField("children").getCount()):
            child = self.getFromDef("PATH").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_path_nodes.append(child)
            self.all_path_nodes_starting_positions.append(child.getField("translation").getSFVec3f())

        self.current_difficulty = {}
        self.number_of_obstacles = 0  # The number of obstacles to use, set from set_difficulty method
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)

        # Path to target stuff
        self.path_to_target = []  # The map cells of the path
        # The min and max (manhattan) distances of the target length allowed, set from set_difficulty method
        self.min_target_dist = 1
        self.max_target_dist = 1

    def set_difficulty(self, difficulty_dict):
        self.current_difficulty = difficulty_dict
        self.number_of_obstacles = difficulty_dict["number_of_obstacles"]
        self.min_target_dist = difficulty_dict["min_target_dist"]
        self.max_target_dist = difficulty_dict["max_target_dist"]
        print("Changed difficulty to:", difficulty_dict)

    def get_action_mask(self):
        mask = [True for _ in range(self.action_space.n)]
        # Mask backward action by default
        mask[3] = False

        # if self.get_ds_values_key() in self.action_masks.keys():
        #     # Mask any action that led to a collision by looking in the dynamically updated action_masks
        #     mask[self.action_masks[self.get_ds_values_key()]] = False
        #     # print(f"Masked action {self.action_names[self.action_masks[self.get_ds_values_key()]]}, mask: {mask}")
        if self.current_dist_sensors[0] < 1.0 or self.current_dist_sensors[1] < 3.0:
            # Mask turn left action when we get a minimum value on the left-most sensors
            mask[1] = False
        if self.current_dist_sensors[-1] < 1.0 or self.current_dist_sensors[-2] < 3.0:
            # Mask turn right action when we get a minimum value on the right-most sensors
            mask[2] = False

        # Unmask backward action if any sensor is reading a small value
        for i in range(1, len(self.current_dist_sensors) - 1):
            if self.current_dist_sensors[i] < self.dist_sensors_threshold:
                mask[3] = True
                break

        # Mask forward action when there is a reading below a threshold in any of the forward-facing sensors
        # to avoid unnecessary collisions
        forward_facing_sensor_thresholds = [0.0, 3.0, 5.0, 4.5, 3.5, 1.5, 1.0, 1.5, 3.5, 4.5, 5.0, 3.0, 0.0]
        for i in range(1, len(self.current_dist_sensors) - 1):
            if self.current_dist_sensors[i] < forward_facing_sensor_thresholds[i]:
                mask[0] = False
                break
        return mask

    def get_observations(self, action=None):
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
        # Add distance, angle, distance change, angle change
        obs = [normalize_to_range(self.current_tar_d, 0.0, self.max_target_distance, 0.0, 1.0, clip=True),
               normalize_to_range(self.current_tar_a, -np.pi, np.pi, -1.0, 1.0, clip=True),
               normalize_to_range(self.previous_tar_d - self.current_tar_d, -0.0013, 0.0013, -1.0, 1.0,
                                  clip=True),
               normalize_to_range(abs(self.previous_tar_a) - abs(self.current_tar_a), -0.0183, 0.0183, -1.0, 1.0,
                                  clip=True)]
        if self.add_action_to_obs:
            # Add action one-hot
            action_one_hot = [0.0 for _ in range(self.action_space.n)]
            try:
                action_one_hot[action] = 1.0
            except IndexError:
                pass
            obs.extend(action_one_hot)

        # Add distance sensor values
        ds_values = []
        for i in range(len(self.distance_sensors)):
            ds_values.append(round(normalize_to_range(self.current_dist_sensors[i], 0, self.ds_max[i], 1.0, 0.0), 2))
        obs.extend(ds_values)

        self.obs_memory = self.obs_memory[1:]  # Drop oldest
        self.obs_memory.append(obs)  # Add the latest observation

        # Add the latest observations based on self.window_latest_dense
        dense_obs = ([self.obs_memory[i] for i in range(len(self.obs_memory) - 1,
                                                        len(self.obs_memory) - 1 - self.window_latest_dense, -1)])

        diluted_obs = []
        counter = 0
        for j in range(len(self.obs_memory) - 2 - self.window_latest_dense, 0, -1):
            counter += 1
            if counter >= self.observation_counter_limit - 1:
                diluted_obs.append(self.obs_memory[j])
                counter = 0
        self.obs_list = []
        for single_obs in diluted_obs:
            for item in single_obs:
                self.obs_list.append(item)
        for single_obs in dense_obs:
            for item in single_obs:
                self.obs_list.append(item)

        return self.obs_list

    def get_reward(self, action):
        # Reward for decreasing distance to the target
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d
        dist_tar_reward = round(normalize_to_range(self.previous_tar_d - self.current_tar_d,
                                                   -0.0013, 0.0013, -1.0, 1.0, clip=True), 2)

        # Reward for decreasing angle to the target
        ang_tar_reward = 0.0
        if (abs(self.previous_tar_a) - abs(self.current_tar_a)) > 0.001:
            ang_tar_reward = 1.0
        elif (abs(self.previous_tar_a) - abs(self.current_tar_a)) < -0.001:
            ang_tar_reward = -1.0

        # Reward for reaching the target
        reach_tar_reward = 0.0
        if self.current_tar_d < self.on_target_threshold:
            reach_tar_reward = 1.0
            self.trigger_done = True  # Terminate episode

        # Reward for distance sensors values
        dist_sensors_reward = 0
        for i in range(len(self.distance_sensors)):
            if self.current_dist_sensors[i] < self.dist_sensors_threshold:
                dist_sensors_reward = -1.0  # If any sensor is under threshold assign penalty
                break

        # Check if the robot has collided with anything, assign negative reward
        collision_reward = 0.0
        if self.touch_sensor.getValue() == 1.0:  # NOQA
            self.collisions_counter += 1
            if self.collisions_counter >= self.reset_on_collisions - 1:
                self.trigger_done = True
                self.collisions_counter = 0
            # Add action that lead to the first collision to action mask, only if an obstacle is detected
            # through the sensors
            # if sum(self.current_dist_sensors) < sum(self.ds_max) and self.collisions_counter == 1:
            #     self.action_masks[self.get_ds_values_key()] = self.previous_action
            #     print(f"Added new action mask for ds: {self.get_ds_values_key()}, "
            #           f"action: {self.action_names[self.previous_action]}.")
            collision_reward = -1.0
        self.previous_action = action

        # Assign a penalty for each step
        time_penalty = -1.0

        ################################################################################################################
        # Total reward calculation
        weighted_dist_tar_reward = round(self.reward_weight_dict["dist_tar"] * dist_tar_reward, 4)
        weighted_ang_tar_reward = round(self.reward_weight_dict["ang_tar"] * ang_tar_reward, 4)
        weighted_dist_sensors_reward = round(self.reward_weight_dict["dist_sensors"] * dist_sensors_reward, 4)
        weighted_reach_tar_reward = round(self.reward_weight_dict["tar_reach"] * reach_tar_reward, 4)
        weighted_collision_reward = round(self.reward_weight_dict["collision"] * collision_reward, 4)
        weighted_time_penalty = round(self.reward_weight_dict["time_penalty_weight"] * time_penalty, 4)

        if weighted_dist_sensors_reward != 0:
            weighted_dist_tar_reward = weighted_dist_tar_reward * self.tar_d_weight_multiplier
            weighted_ang_tar_reward = weighted_ang_tar_reward * self.tar_a_weight_multiplier

        # Calculate normed reward and add it to sum to use it as metric
        weights_sum = sum(self.reward_weight_dict.values())
        weights_normed = {}
        for key, val in self.reward_weight_dict.items():
            weights_normed[key] = (val / weights_sum)
        self.sum_normed_reward += (weights_normed["dist_tar"] * dist_tar_reward +
                                   weights_normed["ang_tar"] * ang_tar_reward +
                                   weights_normed["dist_sensors"] * dist_sensors_reward +
                                   weights_normed["tar_reach"] * collision_reward +
                                   weights_normed["collision"] * reach_tar_reward +
                                   weights_normed["time_penalty_weight"] * time_penalty)

        # Add various weighted rewards together
        reward = (weighted_dist_tar_reward + weighted_ang_tar_reward + weighted_dist_sensors_reward +
                  weighted_collision_reward + weighted_reach_tar_reward + weighted_time_penalty)

        if self.verbose:
            print(f"tar dist : {weighted_dist_tar_reward}")
            print(f"tar ang  : {weighted_ang_tar_reward}")
            print(f"tar stop : {weighted_reach_tar_reward}")
            print(f"sens dist: {weighted_dist_sensors_reward}")
            print(f"col obst : {weighted_collision_reward}")
            print(f"time pen : {weighted_time_penalty}")
            print(f"final reward: {reward}")
            print("-------")

        if self.just_reset:
            self.just_reset = False
            return 0.0
        else:
            return reward

    def reset_sum_reward(self):
        self.sum_normed_reward = 0.0

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
        Resets the simulation physics and objects and re-initializes robot and target positions.
        """
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.obs_list = self.get_default_observation()
        # Reset path
        self.path_to_target = None
        self.motor_speeds = [0.0, 0.0]
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.collisions_counter = 0

        # Set robot random rotation
        self.robot.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

        # Randomize obstacles and target
        if self.current_difficulty["type"] == "random":
            while True:
                # Randomize robot and obstacle positions
                self.randomize_map("random")
                self.simulationResetPhysics()
                # Set the target in a valid position and find a path to it
                # and repeat until a reachable position has been found for the target
                self.path_to_target = self.get_random_path(add_target=True)
                if self.path_to_target is not None:
                    self.path_to_target = self.path_to_target[1:]  # Remove starting node
                    break
        elif self.current_difficulty["type"] == "box":
            while True:
                max_distance_allowed = 1
                # Randomize robot and obstacle positions
                self.randomize_map("box", max_distance_allowed=max_distance_allowed)
                self.simulationResetPhysics()
                # Set the target in a valid position and find a path to it
                # and repeat until a reachable position has been found for the target
                self.path_to_target = self.get_random_path(add_target=False)
                if self.path_to_target is not None:
                    self.path_to_target = self.path_to_target[1:]  # Remove starting node
                    break
                max_distance_allowed += 1

        elif self.current_difficulty["type"] == "corridor":
            while True:
                max_distance_allowed = 1
                # Randomize robot and obstacle positions
                self.randomize_map("corridor")
                self.simulationResetPhysics()
                # Set the target in a valid position and find a path to it
                # and repeat until a reachable position has been found for the target
                self.path_to_target = self.get_random_path(add_target=False)
                if self.path_to_target is not None:
                    self.path_to_target = self.path_to_target[1:]  # Remove starting node
                    break
                max_distance_allowed += 1
        self.place_path(self.path_to_target)
        self.just_reset = True

        # Finally, reset viewpoint, so it plays nice
        self.viewpoint.getField("position").setSFVec3f(self.viewpoint_position)
        self.viewpoint.getField("orientation").setSFRotation(self.viewpoint_orientation)

        return self.obs_list

    def get_default_observation(self):
        """
        Basic get_default_observation implementation that returns a zero vector
        in the shape of the observation space.
        :return: A list of zeros in shape of the observation space
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def update_current_metrics(self):
        # Save previous values
        self.previous_tar_d = self.current_tar_d
        self.previous_tar_a = self.current_tar_a
        self.previous_dist_sensors = self.current_dist_sensors

        # Target distance and angle
        self.current_tar_d = get_distance_from_target(self.robot, self.target)
        self.current_tar_a = get_angle_from_target(self.robot, self.target)

        # Get all distance sensor values
        self.current_dist_sensors = []  # Values are in range [0, self.ds_max]
        for ds in self.distance_sensors:
            self.current_dist_sensors.append(ds.getValue())  # NOQA

    def step(self, action):
        action = self.apply_action(action)

        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        self.update_current_metrics()

        obs = self.get_observations(action)
        rew = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        return (
            obs,
            rew,
            done,
            info
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
        key = self.keyboard.getKey()
        if key == ord("O"):
            print(self.obs_list)
        if key == ord("R"):
            print(self.get_reward(action))
        if self.manual_control:
            action = 4
        gas = 0.0
        wheel = 0.0
        if key == ord("W"):
            action = 0
        if key == ord("A"):
            action = 1
        if key == ord("D"):
            action = 2
        if key == ord("X"):
            action = 3
        if key == ord("S"):
            action = 4
        if key == Keyboard.CONTROL + ord("W"):
            action = 5
        if key == Keyboard.CONTROL + ord("A"):
            action = 6
        if key == Keyboard.CONTROL + ord("D"):
            action = 7
        if key == Keyboard.CONTROL + ord("X"):
            action = 8
        if key == ord("Q"):
            action = 9
        if key == ord("E"):
            action = 10
        if key == ord("Z"):
            action = 11
        if key == ord("C"):
            action = 12

        if action == 0:  # Move forward
            gas = 1.0
            wheel = 0.0
        elif action == 1:  # Turn left
            gas = 0.0
            wheel = -1.0
        elif action == 2:  # Turn right
            gas = 0.0
            wheel = 1.0
        elif action == 3:  # Move backwards
            gas = -1.0
            wheel = 0.0
        elif action == 4:  # Stop
            gas = 0.0
            wheel = 0.0
        elif action == 5:  # Move forward fast
            gas = 4.0
            wheel = 0.0
        elif action == 6:  # Move left fast
            gas = 0.0
            wheel = -4.0
        elif action == 7:  # Move right fast
            gas = 0.0
            wheel = 4.0
        elif action == 8:  # Move backwards fast
            gas = -4.0
            wheel = 0.0
        elif action == 9:  # Move forward-left fast
            gas = 4.0
            wheel = -4.0
        elif action == 10:  # Move forward-right fast
            gas = 4.0
            wheel = 4.0
        elif action == 11:  # Move backwards-left fast
            gas = -4.0
            wheel = 4.0
        elif action == 12:  # Move backwards-right fast
            gas = -4.0
            wheel = -4.0

        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        motor_speeds = [0.0, 0.0]
        motor_speeds[0] = gas + wheel
        motor_speeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        motor_speeds = np.clip(motor_speeds, -4, 4)

        # Apply motor speeds
        self.set_velocity(motor_speeds[0], motor_speeds[1])
        return action

    def get_ds_values_key(self):
        return str([int(self.current_dist_sensors[i]) for i in range(len(self.current_dist_sensors))])

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

    def remove_objects(self):
        """
        Removes objects from arena.
        """
        for object_node, starting_pos in zip(self.all_obstacles, self.all_obstacles_starting_positions):
            object_node.getField("translation").setSFVec3f(starting_pos)
            object_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        for path_node, starting_pos in zip(self.all_path_nodes, self.all_path_nodes_starting_positions):
            path_node.getField("translation").setSFVec3f(starting_pos)
            path_node.getField("rotation").setSFRotation([0, 0, 1, 0])

    def randomize_map(self, type_="random", max_distance_allowed=2):
        """
        TODO docstring
        """
        self.remove_objects()
        self.map.empty()
        robot_z = 0.0399261

        if type_ == "random":
            self.map.add_random(self.robot, robot_z)  # Add robot in a random position
            for obs_node in random.sample(self.all_obstacles, self.number_of_obstacles):
                self.map.add_random(obs_node)
                obs_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
        elif type_ == "box":
            self.map.add_random(self.robot, robot_z)  # Add robot in a random position
            robot_coordinates = self.map.find_by_name("robot")
            # Keep trying to add target near robot at specified min-max distances
            while not self.map.add_near(robot_coordinates[0], robot_coordinates[1],
                                        self.target,
                                        min_distance=self.min_target_dist, max_distance=self.max_target_dist):
                pass
            # Insert obstacles around target
            target_pos = self.map.find_by_name("target")[0:2]
            for obs_node in random.sample(self.all_obstacles, self.number_of_obstacles):
                while not self.map.add_near(target_pos[0], target_pos[1], obs_node, min_distance=1,
                                            max_distance=max_distance_allowed):
                    max_distance_allowed += 1
                obs_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
        elif type_ == "corridor":
            # Add robot to starting position
            self.map.add_cell((self.map_width - 1) // 2, self.map_height - 1, self.robot, robot_z)
            robot_coordinates = [(self.map_width - 1) // 2, self.map_height - 1]
            # Limit the provided min, max target distances
            if self.max_target_dist > self.map_height - 1:
                print(f"max_target_dist set out of range, setting to: {min(self.max_target_dist, self.map_height - 1)}")
            if self.min_target_dist > self.map_height - 1:
                print(f"min_target_dist set out of range, setting to: {min(self.min_target_dist, self.map_height - 1)}")
            # Get true min max target positions
            min_target_pos = self.map_height - 1 - min(self.max_target_dist, self.map_height - 1)
            max_target_pos = self.map_height - 1 - min(self.min_target_dist, self.map_height - 1)
            if min_target_pos == max_target_pos:
                target_y = min_target_pos
            else:
                target_y = random.randint(min_target_pos, max_target_pos)
            # Finally add target
            self.map.add_cell(robot_coordinates[0], target_y, self.target)

            # If there is space between target and robot, add obstacles
            if abs(robot_coordinates[1] - target_y) > 1:
                # We add two obstacles on each row between the target and robot so there is one free cell for the path
                # To generate the obstacle placements within the corridor, we need to make sure that there is
                # a free path within the corridor that leads from one row to the next.
                # This means we need to avoid the case where there's a free place in the first column and on the next
                # row the free place is in the third row
                def add_two_obstacles():
                    col_choices = [robot_coordinates[0] + i for i in range(-1, 2, 1)]
                    random_col_1_ = random.choice(col_choices)
                    col_choices.remove(random_col_1_)
                    random_col_2_ = random.choice(col_choices)
                    col_choices.remove(random_col_2_)
                    return col_choices[0], random_col_1_, random_col_2_

                max_obstacles = (abs(robot_coordinates[1] - target_y) - 1) * 2
                random_sample = random.sample(self.all_obstacles, min(max_obstacles, self.number_of_obstacles))
                prev_free_col = 0
                for row_coord, obs_node_index in \
                        zip(range(target_y + 1, robot_coordinates[1]), range(0, len(random_sample), 2)):
                    # For each row between the robot and the target, add 2 obstacles
                    if prev_free_col == 0:
                        # If previous free column is the center one, any positions for the new row are valid
                        prev_free_col, random_col_1, random_col_2 = add_two_obstacles()
                    else:
                        # If previous free column is not the center one, then the new free one cannot be
                        # on the other side
                        current_free_col, random_col_1, random_col_2 = add_two_obstacles()
                        while abs(prev_free_col - current_free_col) == 2:
                            current_free_col, random_col_1, random_col_2 = add_two_obstacles()
                        prev_free_col = current_free_col
                    self.map.add_cell(random_col_1, row_coord, random_sample[obs_node_index])
                    random_sample[obs_node_index].getField("rotation"). \
                        setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
                    self.map.add_cell(random_col_2, row_coord, random_sample[obs_node_index + 1])
                    random_sample[obs_node_index + 1].getField("rotation"). \
                        setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

            # Abuse the grid map and add wall objects as placeholder to limit path finding within the corridor
            for row_coord in range(target_y + 1, robot_coordinates[1]):
                self.map.add_cell(robot_coordinates[0] - 2, row_coord, self.walls[0])
                self.map.add_cell(robot_coordinates[0] + 2, row_coord, self.walls[1])
            new_position = self.walls_starting_positions[0]
            new_position[0] = -0.75
            self.walls[0].getField("translation").setSFVec3f(new_position)
            new_position = self.walls_starting_positions[1]
            new_position[0] = 0.75
            self.walls[1].getField("translation").setSFVec3f(new_position)

    def get_random_path(self, add_target=True):
        """
        TODO docstring
        """
        robot_coordinates = self.map.find_by_name("robot")
        if add_target:
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

    def export_parameters(self, path,
                          net_arch, gamma, target_kl, vf_coef, ent_coef,
                          difficulty_dict, maximum_episode_steps):
        import json
        param_dict = {"experiment_description": self.experiment_desc,
                      "maximum_episode_steps": maximum_episode_steps,
                      "add_action_to_obs": self.add_action_to_obs,
                      "window_latest_dense": self.window_latest_dense,
                      "window_older_diluted": self.window_older_diluted,
                      "max_ds_range": self.ds_max[0],
                      "on_target_threshold": self.on_target_threshold,
                      "rewards_weights": self.reward_weight_dict,
                      "map_width": self.map_width, "map_height": self.map_height, "cell_size": self.cell_size,
                      "difficulty": difficulty_dict,
                      "ppo_params": {
                          "net_arch": net_arch,
                          "gamma": gamma,
                          "target_kl": target_kl,
                          "vf_coef": vf_coef,
                          "ent_coef": ent_coef,
                      }
                      }
        with open(path, 'w') as fp:
            json.dump(param_dict, fp, indent=4)


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

    def add_cell(self, x, y, node, z=None):
        if self.grid[y][x] is None and self.is_in_range(x, y):
            self.grid[y][x] = node
            if z is None:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], node.getPosition()[2]])
            else:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], z])
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

    def add_random(self, node, z=None):
        x = random.randint(0, len(self.grid[0]) - 1)
        y = random.randint(0, len(self.grid) - 1)
        if self.grid[y][x] is None:
            return self.add_cell(x, y, node, z=z)
        else:
            self.add_random(node, z=z)

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
