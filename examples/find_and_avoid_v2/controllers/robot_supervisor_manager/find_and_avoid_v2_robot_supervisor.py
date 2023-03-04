import random
from warnings import warn
import numpy as np
from gym.spaces import Box, Discrete
from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range, get_distance_from_target, get_angle_from_target
from controller import Supervisor, Keyboard


class FindAndAvoidV2RobotSupervisor(RobotSupervisorEnv):
    """
    TODO *Problem description*
    :param description: A description that can be saved in an exported file
    :type description: str
    :param maximum_episode_steps: The maximum steps per episode before timeout reset
    :type maximum_episode_steps: int
    :param step_window: How many steps of observations to add in the observation window, defaults to 1
    :type step_window: int, optional
    :param seconds_window: How many seconds of observations to add in the observation window, defaults to 1
    :type seconds_window: int, optional
    :param add_action_to_obs: Whether to add the latest action one-hot vector to the observation, defaults to True
    :type add_action_to_obs: bool, optional
    :param max_ds_range: The maximum range of the distance sensors in cm, defaults to 100.0
    :type max_ds_range: float, optional
    :param reset_on_collisions: On how many steps of collisions to reset, defaults to 0
    :type reset_on_collisions: int, optional
    :param manual_control: Whether to override agent actions with user keyboard control, defaults to False
    :type manual_control: bool, optional
    :param on_target_threshold: The threshold under which the robot is considered on target, defaults to 0.1
    :type on_target_threshold: float, optional
    :param dist_sensors_threshold: The distance sensor value threshold under which masking occurs and ds rewards are
        calculated, defaults to 10.0
    :type dist_sensors_threshold: float, optional
    :param ds_type: The type of distance sensors to use, can be either "generic" or "sonar", defaults to "generic"
    :type ds_type: str, optional
    :param ds_noise: The percentage of gaussian noise to add to the distance sensors, defaults to 0.05
    :type ds_noise: float, optional
    :param tar_d_weight_multiplier: The multiplier to apply on the target distance reward, when the distance sensors
        values are under the threshold, defaults to 1.0
    :type tar_d_weight_multiplier: float, optional
    :param tar_a_weight_multiplier: The multiplier to apply on the target angle reward, when the distance sensors
        values are under the threshold, defaults to 1.0
    :type tar_a_weight_multiplier: float, optional
    :param target_distance_weight: The target distance reward weight, defaults to 1.0
    :type target_distance_weight: float, optional
    :param tar_angle_weight: The target angle reward weight, defaults to 1.0
    :type tar_angle_weight: float, optional
    :param dist_sensors_weight: The distance sensors reward weight, defaults to 1.0
    :type dist_sensors_weight: float, optional
    :param smoothness_weight: The smoothness reward weight, defaults to 1.0
    :type smoothness_weight: float, optional
    :param tar_reach_weight: The target reach reward weight, defaults to 1.0
    :type tar_reach_weight: float, optional
    :param collision_weight: The collision reward weight, defaults to 1.0
    :type collision_weight: float, optional
    :param time_penalty_weight: The time penalty reward weight, defaults to 1.0
    :type time_penalty_weight: float, optional
    :param map_width: The map width, defaults to 7
    :type map_width: int, optional
    :param map_height: The map height, defaults to 7
    :type map_height: int, optional
    :param cell_size: The cell size, defaults to None, [0.5, 0.5]
    :type cell_size: list, optional
    :param seed: The random seed, defaults to None
    :type seed: int, optional
    """

    def __init__(self, description, maximum_episode_steps, step_window=1, seconds_window=0, add_action_to_obs=True,
                 max_ds_range=100.0, reset_on_collisions=0, manual_control=False,
                 on_target_threshold=0.1, dist_sensors_threshold=10.0, ds_type="generic", ds_noise=0.05,
                 tar_d_weight_multiplier=1.0, tar_a_weight_multiplier=1.0,
                 target_distance_weight=1.0, tar_angle_weight=1.0, dist_sensors_weight=1.0, smoothness_weight=1.0,
                 tar_reach_weight=1.0, not_reach_weight=1.0, collision_weight=1.0, time_penalty_weight=1.0,
                 map_width=7, map_height=7, cell_size=None, seed=None):
        super().__init__()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.experiment_desc = description
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
        # Actions: increase left motor speed, increase right motor speed,
        # decrease left motor speed, decrease right motor speed, keep same speeds (no action)
        self.action_space = Discrete(5)

        self.add_action_to_obs = add_action_to_obs
        self.step_window = step_window
        self.seconds_window = seconds_window
        self.obs_list = []
        # Set up observation low values
        # Distance to target, angle to target, distance change, angle change, motor speed left, motor speed right, touch
        single_obs_low = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
        # Add action one-hot vector
        if self.add_action_to_obs:
            single_obs_low.extend([0.0 for _ in range(self.action_space.n)])
        # Append distance sensor values
        single_obs_low.extend([0.0 for _ in range(self.number_of_distance_sensors)])

        # Set up corresponding observation high values
        single_obs_high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if self.add_action_to_obs:
            single_obs_high.extend([1.0 for _ in range(self.action_space.n)])
        single_obs_high.extend([1.0 for _ in range(self.number_of_distance_sensors)])

        # Expand sizes depending on step window and seconds window
        self.single_obs_size = len(single_obs_low)
        obs_low = []
        obs_high = []
        for _ in range(self.step_window + self.seconds_window):
            obs_low.extend(single_obs_low)
            obs_high.extend(single_obs_high)
            self.obs_list.extend([0.0 for _ in range(self.single_obs_size)])
        # Memory is used for creating the windows in get_observation()
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.seconds_window * int(np.ceil(1000 / self.timestep))) +
                                          self.step_window)]
        self.observation_counter_limit = int(np.ceil(1000 / self.timestep))
        self.observation_counter = self.observation_counter_limit

        # Finally initialize space
        self.observation_space = Box(low=np.array(obs_low),
                                     high=np.array(obs_high),
                                     dtype=np.float64)

        # Set up sensors
        self.distance_sensors = []
        self.ds_max = []
        # self.ds_min = []
        self.dist_sensors_threshold = dist_sensors_threshold
        self.ds_type = ds_type
        self.ds_noise = ds_noise
        # self.ds_nodes = []
        # Loop through the ds_group node to set max sensor values and initialize the devices and set the type
        robot_children = self.robot.getField("children")
        for childNodeIndex in range(robot_children.getCount()):
            robot_child = robot_children.getMFNode(childNodeIndex)  # NOQA
            if robot_child.getTypeName() == "Group":
                ds_group = robot_child.getField("children")
                for i in range(self.number_of_distance_sensors):
                    self.distance_sensors.append(self.getDevice(f"distance sensor({str(i)})"))
                    self.distance_sensors[-1].enable(self.timestep)  # NOQA
                    ds_node = ds_group.getMFNode(i)
                    ds_node.getField("lookupTable").setMFVec3f(4, [max_ds_range / 100.0, max_ds_range])
                    ds_node.getField("lookupTable").setMFVec3f(3, [0.75 * max_ds_range / 100.0, 0.75 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(2, [0.5 * max_ds_range / 100.0, 0.5 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(1, [0.25 * max_ds_range / 100.0, 0.25 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("type").setSFString(self.ds_type)
                    # self.ds_nodes.append(ds_node)
                    self.ds_max.append(max_ds_range)  # NOQA
                    # self.ds_min.append(ds_node.getField("lookupTable").getMFVec3f(0)[0] * 100)  # NOQA

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
        self.target.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

        # Set up misc
        self.on_target_threshold = on_target_threshold  # Threshold that defines whether robot is considered "on target"
        # Various metrics, for the current step and the previous step
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.current_rotation = 0.0
        self.previous_rotation = 0.0
        self.current_rotation_change = 0.0
        self.previous_rotation_change = 0.0

        # Dictionary holding the weights for the various reward components
        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": tar_angle_weight,
                                   "dist_sensors": dist_sensors_weight, "tar_reach": tar_reach_weight,
                                   "not_reach_weight": not_reach_weight, "collision": collision_weight,
                                   "time_penalty_weight": time_penalty_weight, "smoothness_weight": smoothness_weight}
        self.tar_d_weight_multiplier = tar_d_weight_multiplier
        self.tar_a_weight_multiplier = tar_a_weight_multiplier
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

        # Various metrics and stuff
        self.current_timestep = 0
        self.maximum_episode_steps = maximum_episode_steps
        self.done_reason = ""  # Used to print the reason the episode is done while testing
        self.reset_count = -1
        self.reach_target_count = 0
        self.collision_termination_count = 0
        self.timeout_count = 0
        self.initial_target_distance = 0.0
        self.initial_target_angle = 0.0
        self.min_distance_reached = float("inf")

    def set_difficulty(self, difficulty_dict, key=None):
        """
        Sets the difficulty and corresponding variables with a difficulty dictionary provided.

        :param difficulty_dict: A dictionary containing the difficulty information, e.g.
            {"type": "t", "number_of_obstacles": X, "min_target_dist": Y, "max_target_dist": Z}, where type can be
            "random" or "corridor".
        :type difficulty_dict: dict
        :param key: The key of the difficulty dictionary, if provided, the method prints it along with the
            difficulty dict, defaults to None
        :type key: str, optional
        """
        self.current_difficulty = difficulty_dict
        self.number_of_obstacles = difficulty_dict["number_of_obstacles"]
        self.min_target_dist = difficulty_dict["min_target_dist"]
        self.max_target_dist = difficulty_dict["max_target_dist"]
        if key is not None:
            print(f"Changed difficulty to: {key}, {difficulty_dict}")
        else:
            print("Changed difficulty to:", difficulty_dict)

    def get_action_mask(self):
        """
        Returns the mask for the current state. The mask is a list of bools where each element corresponds to an
        action, and if the bool is False the corresponding action is masked, i.e. disallowed.
        - Backward movement is disallowed unless there is an obstacle detected through the distance sensors, under the
        distance sensor threshold.
        - Left and right movement is disallowed depending on the values of the side-most distance sensors.
        - Forward movement is disallowed when the forward-facing sensors are under specific thresholds for each.

        :return: The action mask list of bools
        :rtype: list of booleans
        """
        # Actions: increase left motor speed, increase right motor speed,
        # decrease left motor speed, decrease right motor speed, keep same speeds (no action)
        mask = [True for _ in range(self.action_space.n)]
        # Mask decrease actions that will cause the agent to move backwards by default
        if self.motor_speeds[0] <= 0.0 and self.motor_speeds[1] <= 0.0:
            mask[2] = False
            mask[3] = False

        if self.current_dist_sensors[0] < 1.0 or self.current_dist_sensors[1] < 3.0:
            # Mask increase right action when we get a minimum value on the left-most sensors
            mask[1] = False
        if self.current_dist_sensors[-1] < 1.0 or self.current_dist_sensors[-2] < 3.0:
            # Mask increase left action when we get a minimum value on the right-most sensors
            mask[0] = False

        # Unmask backward action if any sensor is reading a small value
        for i in range(len(self.current_dist_sensors)):
            if self.current_dist_sensors[i] < self.dist_sensors_threshold:
                mask[2] = True
                mask[3] = True
                break

        # Mask increasing of speed actions when there is a reading below a threshold in any of the
        # forward-facing sensors to avoid unnecessary collisions
        forward_facing_sensor_thresholds = [0.0, 3.0, 5.0, 4.5, 3.5, 1.5, 1.0, 1.5, 3.5, 4.5, 5.0, 3.0, 0.0]
        for i in range(1, len(self.current_dist_sensors) - 1):
            if self.current_dist_sensors[i] < forward_facing_sensor_thresholds[i]:
                mask[0] = False
                mask[1] = False
                break
        return mask

    def get_observations(self, action=None):
        """
        This method returns the observation list of the agent.
        A single observation consists of the distance and angle to the target, the latest change of the distance and
        angle to target, the current motor speeds, the touch sensor value, the latest action represented by
        a one-hot vector, and finally the distance sensor values.

        All values are normalized in their respective ranges, where appropriate:
        - Distance is normalized to [0.0, 1.0]
        - Angle is normalized to [-1.0, 1.0]
        - Distance and angle change to [-1.0, 1.0]
        - Motor speeds are already constrained within [-1.0, 1.0]
        - Touch sensor value can only been 0 or 1
        - Distance sensor values are normalized to [1.0, 0.0]
          This is done so the input gets a large activation value when the sensor returns
          small values, i.e. an obstacle is close.

        Note that the normalization of the distance and angle change values are based on the maximum robot
        movement speed that is defined by the maximum motor speeds [-1, 1].

        All observations are held in a memory (self.obs_memory) and the current observation is augmented with
        self.step_window steps of the latest single observations and with self.seconds_window seconds of
        observations. This means that for self.step_window=2 and self.seconds_window=2, the observation
        is the latest two single observations plus an observation from 1 second in the past and an observation
        from 2 seconds in the past.

        :param action: The latest action, defaults to None to match signature of parent method
        :type action: int, optional
        :return: Observation list
        :rtype: list
        """
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d
            self.previous_tar_a = self.current_tar_a
        # Add distance, angle, distance change, angle change
        obs = [normalize_to_range(self.current_tar_d, 0.0, self.initial_target_distance, 0.0, 1.0, clip=True),
               normalize_to_range(self.current_tar_a, -np.pi, np.pi, -1.0, 1.0, clip=True),
               round(normalize_to_range(self.previous_tar_d - self.current_tar_d, -0.00128, 0.00128, -1.0, 1.0,
                                        clip=True), 4),
               round(normalize_to_range(abs(self.previous_tar_a) - abs(self.current_tar_a), -0.0183, 0.0183, -1.0, 1.0,
                                        clip=True), 4),
               self.motor_speeds[0], self.motor_speeds[1], self.touch_sensor.getValue()]  # NOQA

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
            ds_values.append(normalize_to_range(self.current_dist_sensors[i], 0, self.ds_max[i], 1.0, 0.0))
        obs.extend(ds_values)

        self.obs_memory = self.obs_memory[1:]  # Drop oldest
        self.obs_memory.append(obs)  # Add the latest observation

        # Add the latest observations based on self.step_window
        dense_obs = ([self.obs_memory[i] for i in range(len(self.obs_memory) - 1,
                                                        len(self.obs_memory) - 1 - self.step_window, -1)])

        diluted_obs = []
        counter = 0
        for j in range(len(self.obs_memory) - 1 - self.step_window, 0, -1):
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
        """
        This method calculates the reward. The reward consists of various components that get weighted and added into
        the final reward that gets returned.

        - Distance to target reward is calculated based on the minimum distance to the target achieved for this episode.
        The agent is rewarded with a value of 1.0 every time the it decreases the minimum distance to the target,
        similar to the pit escape problem.
        - Angle to target reward is calculated based on the latest change of the angle to target, if positive and
        over a small threshold, the reward is 1.0, if negative and under a small threshold, the reward is -1.0
        - Reach target reward is 0.0, unless the current distance to target is under the on_target_threshold when it
        becomes 1.0, and reset is triggered (done)
        - Distance sensors reward is 0.0, unless even one of the distance sensors reads a value below the distance
        sensor threshold when it becomes -1.0
        - Obstacle turning reward penalizes ds values getting smaller. For each sensor value changing, a negative reward
        is assigned for decreasing, a positive reward is assigned for increasing. The reward itself is [-1.0, 1.0] for
        each sensor. The final reward is the average reward across all sensors.
        - Collision reward is 0.0, unless the touch sensor detects a collision when it becomes -1.0. It also counts
        the number of collisions and triggers a reset (done) when the set limit reset_on_collisions is reached
        - Time penalty is simply -1.0 for each step

        All these rewards are multiplied with their corresponding weights taken from reward_weight_dict and summed into
        the final reward.

        :param action: The latest action
        :type action: int
        :return: The total reward
        :rtype: float
        """
        # Reward for decreasing distance to the target
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d = self.initial_target_distance
            self.previous_tar_a = self.current_tar_a = self.initial_target_angle

        dist_tar_reward = 0.0
        if self.current_tar_d - self.min_distance_reached < -0.001:
            dist_tar_reward = 1.0
            self.min_distance_reached = self.current_tar_d

        # Reward for decreasing angle to the target
        ang_tar_reward = 0.0
        if abs(self.previous_tar_a) - abs(self.current_tar_a) > 0.01:
            ang_tar_reward = 1.0
        elif abs(self.previous_tar_a) - abs(self.current_tar_a) < -0.01:
            ang_tar_reward = -1.0

        # Reward for reaching the target
        reach_tar_reward = 0.0
        if self.current_tar_d < self.on_target_threshold:
            reach_tar_reward = 1.0 - 0.5 * self.current_timestep / self.maximum_episode_steps
            self.trigger_done = True  # Terminate episode
            self.done_reason = "reached target"

        # Penalty for distance sensors values
        dist_sensors_reward = 0
        for i in range(len(self.distance_sensors)):
            if self.current_dist_sensors[i] < self.dist_sensors_threshold:
                dist_sensors_reward -= 1.0  # If any sensor is under threshold assign penalty
        dist_sensors_reward /= self.number_of_distance_sensors
        dist_sensors_reward = normalize_to_range(dist_sensors_reward, 0.0, -0.5385, 0.0, -1.0)

        # Check if the robot has collided with anything, assign negative reward
        collision_reward = 0.0
        if self.touch_sensor.getValue() == 1.0:  # NOQA
            self.collisions_counter += 1
            if self.collisions_counter >= self.reset_on_collisions - 1 and self.reset_on_collisions != -1:
                self.trigger_done = True
                self.done_reason = "collision"
                self.collisions_counter = 0
            collision_reward = -1.0

        # Assign a penalty for each step
        time_penalty = -1.0

        # Episode will terminate and robot didn't reach the target
        not_reach_reward = 0.0
        if self.current_timestep == self.maximum_episode_steps - 1:
            not_reach_reward = normalize_to_range(get_distance_from_target(self.robot, self.target),
                                                  0.0, self.initial_target_distance, 0.0, -1.0)

        smoothness_reward = 0.0
        if abs(self.current_rotation - self.previous_rotation) > 0.00016928 and dist_sensors_reward == 0.0:
            if not (0.0 < self.current_rotation_change * self.previous_rotation_change or 6.0 < abs(
                    self.current_rotation_change - self.previous_rotation_change) < 7.0):
                smoothness_reward = -1.0

        ################################################################################################################
        # Total reward calculation
        weighted_dist_tar_reward = self.reward_weight_dict["dist_tar"] * dist_tar_reward
        weighted_ang_tar_reward = self.reward_weight_dict["ang_tar"] * ang_tar_reward
        weighted_dist_sensors_reward = self.reward_weight_dict["dist_sensors"] * dist_sensors_reward
        weighted_reach_tar_reward = self.reward_weight_dict["tar_reach"] * reach_tar_reward
        weighted_not_reach_tar_reward = self.reward_weight_dict["not_reach_weight"] * not_reach_reward
        weighted_collision_reward = self.reward_weight_dict["collision"] * collision_reward
        weighted_time_penalty = self.reward_weight_dict["time_penalty_weight"] * time_penalty
        weighted_smoothness_reward = self.reward_weight_dict["smoothness_weight"] * smoothness_reward

        if weighted_dist_sensors_reward != 0:
            weighted_dist_tar_reward = weighted_dist_tar_reward * self.tar_d_weight_multiplier
            weighted_ang_tar_reward = weighted_ang_tar_reward * self.tar_a_weight_multiplier

        # Add various weighted rewards together
        reward = (weighted_dist_tar_reward + weighted_ang_tar_reward + weighted_dist_sensors_reward +
                  weighted_collision_reward + weighted_reach_tar_reward + weighted_not_reach_tar_reward +
                  weighted_time_penalty + weighted_smoothness_reward)

        if self.just_reset:
            self.just_reset = False
            return 0.0
        else:
            return reward

    def is_done(self):
        """
        Episode done triggers from the trigger_done flag which is set in the reward function, when the maximum
        number of collisions is reached or the target is reached.

        :return: Whether the episode is done
        :rtype: bool
        """
        if self.trigger_done:
            return True
        return False

    def reset(self):
        """
        Resets the simulation physics and objects and re-initializes robot and target positions,
        along any other variables that need to be reset to their original values.

        Then it creates the new obstacle map depending on difficulty, and resets the viewpoint.
        """
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))  # NOQA
        self.obs_list = self.get_default_observation()
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.seconds_window * int(np.ceil(1000 / self.timestep))) +
                                          self.step_window)]
        self.observation_counter = self.observation_counter_limit
        # Reset path and various values
        self.current_timestep = 0
        self.trigger_done = False
        self.path_to_target = None
        self.motor_speeds = [0.0, 0.0]
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.current_rotation = 0.0
        self.previous_rotation = 0.0
        self.current_rotation_change = 0.0
        self.previous_rotation_change = 0.0
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
        self.reset_count += 1
        if self.done_reason != "":
            print("Resetting, done reason:", self.done_reason)
        if self.done_reason == "collision":
            self.collision_termination_count += 1
        elif self.done_reason == "reached target":
            self.reach_target_count += 1
        elif self.done_reason == "timeout":
            self.timeout_count += 1
        self.done_reason = ""
        self.initial_target_distance = get_distance_from_target(self.robot, self.target)
        self.initial_target_angle = get_angle_from_target(self.robot, self.target)
        self.min_distance_reached = self.initial_target_distance
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
        """
        Updates any metric that needs to be updated in each step.
        """
        # Save previous values
        self.previous_tar_d = self.current_tar_d
        self.previous_tar_a = self.current_tar_a
        self.previous_dist_sensors = self.current_dist_sensors
        self.previous_rotation = self.current_rotation
        self.previous_rotation_change = self.current_rotation_change

        # Target distance and angle
        self.current_tar_d = get_distance_from_target(self.robot, self.target)
        self.current_tar_a = get_angle_from_target(self.robot, self.target)

        # Get current rotation
        self.current_rotation = abs(self.robot.getField('rotation').getSFRotation()[3] * np.sign(
            self.robot.getField('rotation').getSFRotation()[2]) + np.pi)
        self.current_rotation_change = self.current_rotation - self.previous_rotation

        # Get all distance sensor values
        self.current_dist_sensors = []  # Values are in range [0, self.ds_max]
        for ds in self.distance_sensors:
            self.current_dist_sensors.append(ds.getValue())  # NOQA

    def step(self, action):
        """
        Step override method which slightly modifies the parent.
        It applies the previous action, steps the simulation, updates the metrics with new values and then
        gets the new observation, reward, done flag and info and returns them.

        :param action: The action to perform
        :type action: int
        :return: new observation, reward, done flag, info
        :rtype: tuple
        """
        action = self.apply_action(action)

        if super(Supervisor, self).step(self.timestep) == -1:  # NOQA
            exit()

        self.update_current_metrics()

        obs = self.get_observations(action)
        rew = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        self.current_timestep += 1
        # Timeout
        if self.current_timestep >= self.maximum_episode_steps:
            done = True
            self.done_reason = "timeout"
            info = {"done_reason": self.done_reason}
        return (
            obs,
            rew,
            done,
            info
        )

    def set_maximum_episode_steps(self, new_value):
        """
        This is required to be able to change the value and sb3 registers it.
        """
        self.maximum_episode_steps = new_value

    def apply_action(self, action):
        """
        This method gets an integer action value [0, 1, ...] where each value
        corresponds to an action:
        0: Increase left motor speed
        1: Increase right motor speed
        2: Decrease left motor speed
        3: Decrease right motor speed
        4: No action

        and applies the action by changing and setting the motor speeds.

        This method also incorporates the keyboard control and if the user presses any of the
        control buttons that correspond to more user-friendly actions
        (W: move forward, A: turn left, S: move backwards, D: turn right), it applies and returns no action
        for the agent.

        :param action: The action to execute
        :type action: int
        :return: The action executed
        :rtype: int
        """
        key = self.keyboard.getKey()
        if key == ord("O"):  # Print latest observation
            print(self.obs_memory[-1])
        if key == ord("R"):  # Print latest reward
            print(self.get_reward(action))

        if self.manual_control:
            action = 4
            self.motor_speeds[0] = 0.0
            self.motor_speeds[1] = 0.0
        if key == ord("W"):  # Move forward
            action = 4
            self.motor_speeds[0] = 4.0
            self.motor_speeds[1] = 4.0
        if key == ord("S"):  # Move backward
            action = 4
            self.motor_speeds[0] = -4.0
            self.motor_speeds[1] = -4.0
        if key == ord("A"):  # Turn Left
            action = 4
            self.motor_speeds[0] = -2.0
            self.motor_speeds[1] = 2.0
        if key == ord("D"):  # Turn right
            action = 4
            self.motor_speeds[0] = 2.0
            self.motor_speeds[1] = -2.0

        if action == 0:  # Increase left wheel
            if self.motor_speeds[0] < 1.0:
                self.motor_speeds[0] += 0.25
        elif action == 1:  # Increase right wheel
            if self.motor_speeds[1] < 1.0:
                self.motor_speeds[1] += 0.25
        elif action == 2:  # Decrease left wheel
            if self.motor_speeds[0] > -1.0:
                self.motor_speeds[0] -= 0.25
        elif action == 3:  # Decrease right wheel
            if self.motor_speeds[1] > -1.0:
                self.motor_speeds[1] -= 0.25
        elif action == 4:  # No action
            pass

        # Apply motor speeds
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
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

    def remove_objects(self):
        """
        Removes objects from arena, by setting their original translations and rotations.
        """
        for object_node, starting_pos in zip(self.all_obstacles, self.all_obstacles_starting_positions):
            object_node.getField("translation").setSFVec3f(starting_pos)
            object_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        for path_node, starting_pos in zip(self.all_path_nodes, self.all_path_nodes_starting_positions):
            path_node.getField("translation").setSFVec3f(starting_pos)
            path_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        for wall_node, starting_pos in zip(self.walls, self.walls_starting_positions):
            wall_node.getField("translation").setSFVec3f(starting_pos)
            wall_node.getField("rotation").setSFRotation([0, 0, 1, -1.5708])

    def randomize_map(self, type_="random"):
        """
        Randomizes the obstacles on the map, by first removing all the objects and emptying the grid map.
        Then, based on the type_ argument provided, places the set number of obstacles in various random configurations.

        - "random": places the number_of_obstacles in free positions on the map and randomizes their rotation
        - "corridor": creates a corridor placing the robot at the start and the target at a distance along the corridor.
            It then places the obstacles on each row along the corridor between the target and the robot, making sure
            there is a valid path, i.e. consecutive rows should have free cells either diagonally or in the same column

        :param type_: The type of randomization, either "random" or "corridor", defaults to "random"
        :type type_: str, optional
        """
        self.remove_objects()
        self.map.empty()
        robot_z = 0.0399261  # Custom z value for the robot to avoid physics issues

        if type_ == "random":
            self.map.add_random(self.robot, robot_z)  # Add robot in a random position
            for obs_node in random.sample(self.all_obstacles, self.number_of_obstacles):
                self.map.add_random(obs_node)
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
                # row the free place is in the third row.
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
            new_position = [-0.75,
                            self.walls_starting_positions[0][1],
                            self.walls_starting_positions[0][2]]
            self.walls[0].getField("translation").setSFVec3f(new_position)
            new_position = [0.75,
                            self.walls_starting_positions[1][1],
                            self.walls_starting_positions[1][2]]
            self.walls[1].getField("translation").setSFVec3f(new_position)

    def get_random_path(self, add_target=True):
        """
        Returns a path to the target or None if path is not found. Based on the add_target flag it also places the
        target randomly at a certain manhattan min/max distance to the robot.

        :param add_target: Whether to also add the target before returning the path, defaults to True
        :type add_target: bool, optional
        """
        robot_coordinates = self.map.find_by_name("robot")
        if add_target:
            if not self.map.add_near(robot_coordinates[0], robot_coordinates[1],
                                     self.target,
                                     min_distance=self.min_target_dist, max_distance=self.max_target_dist):
                return None  # Need to re-randomize obstacles as add_near failed
        return self.map.bfs_path(robot_coordinates, self.map.find_by_name("target"))

    def place_path(self, path):
        """
        Place the path nodes (the small deepbots logos) on their proper places depending on the path generated.

        :param path: The path list
        :type path: list
        """
        for p, l in zip(path, self.all_path_nodes):
            self.map.add_cell(p[0], p[1], l)

    def find_dist_to_path(self):
        """
        This method is not currently used. It calculates the closest point and distance to the path,
        returning both.

        :return: The minimum distance to the path and the corresponding closest point on the path
        :rtype: tuple
        """

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
        Returns the reason the episode is done, used only while testing.

        :return: Dictionary containing information
        :rtype: dict
        """
        if self.trigger_done:
            return {"done_reason": self.done_reason}
        else:
            return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render.

        :param mode: defaults to 'human'
        :type mode: str, optional
        """
        print("render() is not used")

    def export_parameters(self, path,
                          net_arch, gamma, gae_lambda, target_kl, vf_coef, ent_coef,
                          difficulty_dict, maximum_episode_steps, n_steps, batch_size):
        """
        Exports all parameters that define the environment/experiment setup.

        :param path: The path to save the export
        :type path: str
        :param net_arch: The network architectures, e.g. dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])
        :type net_arch: dict with two lists
        :param gamma: The gamma value
        :type gamma: float
        :param gae_lambda: The GAE lambda value
        :type gae_lambda: float
        :param target_kl: The target_kl value
        :type target_kl: float
        :param vf_coef: The vf_coef value
        :type vf_coef: float
        :param ent_coef: The ent_coef value
        :type ent_coef: float
        :param difficulty_dict: The difficulty dict
        :type difficulty_dict: dict
        :param maximum_episode_steps: The maximum episode steps before timeout
        :type maximum_episode_steps: int
        :param n_steps: Number of steps between each training session for sb3
        :type n_steps: int
        :param batch_size: The batch size used during training
        :type batch_size: int
        """
        import json
        param_dict = {"experiment_description": self.experiment_desc,
                      "seed": self.seed,
                      "n_steps:": n_steps,
                      "batch_size": batch_size,
                      "maximum_episode_steps": maximum_episode_steps,
                      "add_action_to_obs": self.add_action_to_obs,
                      "step_window": self.step_window,
                      "seconds_window": self.seconds_window,
                      "ds_type": self.ds_type,
                      "ds_noise": self.ds_noise,
                      "max_ds_range": self.ds_max[0],
                      "dist_sensors_threshold": self.dist_sensors_threshold,
                      "tar_d_weight_multiplier": self.tar_d_weight_multiplier,
                      "tar_a_weight_multiplier": self.tar_a_weight_multiplier,
                      "rewards_weights": self.reward_weight_dict,
                      "map_width": self.map_width, "map_height": self.map_height, "cell_size": self.cell_size,
                      "difficulty": difficulty_dict,
                      "ppo_params": {
                          "net_arch": net_arch,
                          "gamma": gamma,
                          "gae_lambda": gae_lambda,
                          "target_kl": target_kl,
                          "vf_coef": vf_coef,
                          "ent_coef": ent_coef,
                      }
                      }
        with open(path, 'w') as fp:
            json.dump(param_dict, fp, indent=4)


class Grid:
    """
    The grid map used to place all objects in the arena and find the paths.

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
