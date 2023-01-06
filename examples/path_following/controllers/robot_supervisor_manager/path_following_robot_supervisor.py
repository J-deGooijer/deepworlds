from deepbots.supervisor import RobotSupervisorEnv
from gym.spaces import Box, Discrete
import numpy as np
from random import uniform
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
        self.observation_space = Box(low=np.array([0.0, -1.0]),
                                     high=np.array([1.0, 1.0]),
                                     dtype=np.float64)
        self.action_space = Discrete(4)

        # Set up various robot components
        self.robot = self.getSelf()

        # try:
        #     self.camera = self.getDevice("camera")
        #     self.camera.enable(self.timestep)  # NOQA
        # except AttributeError:
        #     print("No camera found.")

        try:
            self.left_distance_sensor = self.getDevice("ds_left")
            self.left_distance_sensor.enable(self.timestep)  # NOQA
            self.right_distance_sensor = self.getDevice("ds_right")
            self.right_distance_sensor.enable(self.timestep)  # NOQA
        except AttributeError:
            print("No distance sensors found.")

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
        self.d_min = 0.25  # Minimum distance along the x- or y-axis between the robot and the target. 0.25=25cm
        self.d_max = 1.0  # Maximum distance along the x- or y-axis between the robot and the target. 1.0=100cm
        # The real max distance calculation when target is d_max away in both axes
        self.real_d_max = np.sqrt(2 * (self.d_max * self.d_max))

        self.on_target_threshold = 0.05  # Threshold that defines whether robot is considered "on target"
        self.facing_target_threshold = np.pi / 32  # Threshold on which robot is considered facing the target, π/32~5deg
        self.previous_distance = 0.0
        self.previous_angle = 0.0
        self.on_target_counter = 0
        self.on_target_limit = 400  # The number of steps robot should be on target before the target moves

    def get_observations(self):
        """
        TODO docstring
        :return:
        """
        # Target distance
        tar_d = get_distance_from_target(self.robot, self.target)
        tar_d = round(normalize_to_range(tar_d, 0.0, self.real_d_max, 0.0, 1.0), 2)
        # Angle between robot facing and target
        tar_a = get_angle_from_target(self.robot, self.target)
        tar_a = round(normalize_to_range(tar_a, -np.pi, np.pi, -1.0, 1.0), 2)
        # TODO Add distance sensor observations
        return [tar_d, tar_a]

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
                self.set_random_target_position()
                r += 1000
                self.on_target_counter = 0
            else:
                self.on_target_counter += 1
        # If either distance or angle becomes larger than thresholds, reset counter
        if get_distance_from_target(self.robot, self.target) > self.on_target_threshold or \
                get_angle_from_target(self.robot, self.target, is_abs=True) > self.facing_target_threshold:
            self.on_target_counter = 0

        return r

    def is_done(self):
        """
        Episode is done when the maximum number of steps per episode is reached, which is handled by the RL training
        loop, or when the robot hits an obstacle.
        :return: Whether the episode is done
        :rtype: bool
        """
        # TODO Add collision detection with obstacle to terminate episode
        return False

    def reset(self):
        """
        Resets the simulation using deepbots default reset and re-initializes robot and target positions.
        """
        return_val = super().reset()

        # Set robot random rotation
        self.robot.getField("rotation").setSFRotation([0.0, 0.0, 1.0, uniform(-np.pi, np.pi)])
        # Set random target
        self.set_random_target_position()

        # TODO randomize obstacles

        return return_val

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

    def set_random_target_position(self):
        """
        Sets the target position at a random position around the robot.
        With (0, 0) being the robot, the target can be no closer than d_min in both x and y,
        and no farther than d_max in both x and y.
        """
        robot_position = self.robot.getPosition()
        min_max_diff = self.d_max - self.d_min
        random_x = uniform(-min_max_diff + robot_position[0], min_max_diff + robot_position[0])
        random_y = uniform(-min_max_diff + robot_position[1], min_max_diff + robot_position[1])
        random_x = random_x + np.sign(random_x) * self.d_min
        random_y = random_y + np.sign(random_y) * self.d_min
        self.target_position = [round(random_x, 2), round(random_y, 2)]

        self.target.getField("translation").setSFVec3f([self.target_position[0], self.target_position[1],
                                                        self.target.getPosition()[2]])

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
