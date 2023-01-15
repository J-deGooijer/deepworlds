import math
import numpy as np
import matplotlib.pyplot as plt


def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalizes value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :type value: float
    :param min_val: value's min value, value ∈ [min_val, max_val]
    :type min_val: float
    :param max_val: value's max value, value ∈ [min_val, max_val]
    :type max_val: float
    :param new_min: normalized range min value
    :type new_min: float
    :param new_max: normalized range max value
    :type new_max: float
    :param clip: whether to clip normalized value to new range or not, defaults to False
    :type clip: bool, optional
    :return: normalized value ∈ [new_min, new_max]
    :rtype: float
    """
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max


def plot_data(data, x_label, y_label, plot_title, save=False, save_name=None):
    """
    Uses matplotlib to plot data.

    :param data: List of data to plot
    :type data: list
    :param x_label: Label on x-axis
    :type x_label: str
    :param y_label: Label on y-axis
    :type y_label: str
    :param plot_title: Plot title
    :type plot_title: str
    :param save: Whether to save plot automatically or not, defaults to False
    :type save: bool, optional
    :param save_name: Filename of saved plot, defaults to None
    :type save_name: str, optional
    """
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=plot_title)
    ax.grid()
    if save:
        if save_name is not None:
            fig.savefig(save_name)
        else:
            fig.savefig("figure")
    plt.show()


def get_distance_from_target(robot_node, target_node):
    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    target_coordinate = target_node.getField('translation').getSFVec3f()

    dx = robot_coordinates[0] - target_coordinate[0]
    dy = robot_coordinates[1] - target_coordinate[1]
    distance_from_target = math.sqrt(dx * dx + dy * dy)
    return distance_from_target


def get_angle_from_target(robot_node, target, node_mode=True, is_abs=False):
    """
    Returns the angle between the facing vector of the robot and the target position.
    Explanation can be found here https://math.stackexchange.com/a/14180.
    :param robot_node: The robot Webots node
    :type robot_node: controller.node.Node
    :param target: The target Webots node or position
    :type target: controller.node.Node or [x, y]
    :param node_mode: Whether the target is given as a Webots node
    :type node_mode: bool
    :param is_abs: Whether to return the absolute value of the angle. When True,
    eliminates clockwise, anti-clockwise direction and returns [0, π]
    :type is_abs: bool
    :return: The angle between the facing vector of the robot and the target position
    :rtype: float, [-π, π]
    """
    # The sign of the z-axis is needed to flip the rotation sign, because Webots seems to randomly
    # switch between positive and negative z-axis as the robot rotates.
    robot_angle = robot_node.getField('rotation').getSFRotation()[3] * \
        np.sign(robot_node.getField('rotation').getSFRotation()[2])

    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    if node_mode:
        target_coordinate = target.getField('translation').getSFVec3f()
    else:
        target_coordinate = target

    x_r = (target_coordinate[0] - robot_coordinates[0])
    y_r = (target_coordinate[1] - robot_coordinates[1])

    angle_dif = math.atan2(y_r, x_r)
    angle_dif = angle_dif - robot_angle
    if angle_dif > np.pi:
        angle_dif = angle_dif - (2 * np.pi)
    if angle_dif < -np.pi:
        angle_dif = angle_dif + (2 * np.pi)

    if is_abs:
        angle_dif = abs(angle_dif)

    return angle_dif
