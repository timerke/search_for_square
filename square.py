from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


class Square:

    def __init__(self, x_center: float, y_center: float, length: float, angle: float) -> None:
        """
        :param x_center: coordinate of square center along axis X;
        :param y_center: coordinate of square center along axis Y;
        :param length: square side length;
        :param angle: angle of rotation of square relative to its center.
        """

        self._angle: float = angle
        self._length: float = length
        self._x: float = x_center
        self._y: float = y_center
        self._vertices: List[np.ndarray] = self._create_vertices()

    @property
    def center(self) -> Tuple[float, float]:
        """
        :return: coordinates of square center.
        """

        return self._x, self._y

    @property
    def vertices(self) -> List[np.ndarray]:
        """
        :return: coordinates of vertices of square.
        """

        return self._vertices

    def _create_vertices(self) -> np.ndarray:
        vertices = [[self._x + self._length / 2, self._y + self._length / 2],
                    [self._x - self._length / 2, self._y + self._length / 2],
                    [self._x - self._length / 2, self._y - self._length / 2],
                    [self._x + self._length / 2, self._y - self._length / 2]]
        vertices = [np.array([*point, 1]) for point in vertices]
        shift = np.array([[1, 0, -self._x], [0, 1, -self._y], [0, 0, 1]])
        angle = -self._angle
        rotation = np.array([[np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle), 0],
                             [-np.sin(np.pi / 180 * angle), np.cos(np.pi / 180 * angle), 0],
                             [0, 0, 1]])
        shift_back = np.array([[1, 0, self._x], [0, 1, self._y], [0, 0, 1]])
        return [np.matmul(shift_back, np.matmul(rotation, np.matmul(shift, np.transpose(point)))) for point in vertices]

    def _get_side_lines(self) -> List[Tuple[float, float, float]]:
        """
        :return: list with coefficients (A, B, Y) of lines that make up sides of square.
        """

        lines = []
        for i, vertex in enumerate(self._vertices):
            lines.append(create_line(vertex, self._vertices[i - 1]))
        return lines

    @staticmethod
    def _select_intersection(x: float, y: float, angle: float, intersections: List[np.ndarray]) -> Optional[np.ndarray]:
        distances = [(get_distance_from_point_to_point((x, y), point), point) for point in intersections]
        distances = sorted(distances, key=lambda value: value[0])
        angle = change_angle(angle)
        for _, point in distances:
            if 0 <= angle < np.pi / 2 and point[0] >= x and point[1] >= y:
                return point
            if np.pi / 2 <= angle < np.pi and (point[0] <= x or np.isclose(point[0], x)) and point[1] >= y:
                return point
            if np.pi <= angle < 3 * np.pi / 2 and point[0] <= x and (point[1] <= y or np.isclose(point[1], y)):
                return point
            if 3 * np.pi / 2 <= angle < 2 * np.pi and point[0] >= x and point[1] <= y:
                return point
        return None

    def find_intersection(self, x: float, y: float, angle: float) -> Optional[np.ndarray]:
        """
        :param x: coordinate X of point from which ray is drawn;
        :param y: coordinate Y of point from which ray is drawn;
        :param angle: angle to axis X beam tilt.
        :return: coordinates of point of intersection of a given ray with side of square.
        """

        side_lines = self._get_side_lines()
        line = create_line_for_ray(x, y, angle)
        intersections = []
        for side_line in side_lines:
            intersection = find_intersection(line, side_line)
            if intersection is not None:
                intersections.append(intersection)
        point = self._select_intersection(x, y, angle, intersections)
        return point


def add_error_to_points(points: List[np.ndarray], max_xy_shift: float) -> List[np.ndarray]:
    """
    :param points: list with coordinates of points;
    :param max_xy_shift: maximum shift that coordinates along axes X and Y can have.
    :return: list of coordinates of points with random errors.
    """

    points_with_error = []
    for point in points:
        shift = np.array([max_xy_shift * (1 - 2 * np.random.random()), max_xy_shift * (1 - 2 * np.random.random())])
        point_with_error = point + shift
        points_with_error.append(point_with_error)
    return points_with_error


def calculate_sides(points: List[np.ndarray], length: float, center: Tuple[float, float]) -> np.ndarray:
    """
    :param points: list with approximate coordinates of points on sides of square;
    :param length: square side length;
    :param center: approximate coordinates of center of square.
    :return: list with coefficients (A, B, C) of lines coinciding with sides of optimal square.
    """

    def _get_distances(coefficients: np.ndarray) -> List[float]:
        distances = []
        a_1, a_2, b_1, c_1, c_2 = coefficients
        b_2 = -a_1 * a_2 / b_1
        c_3 = c_1 + length * np.power(a_1**2 + b_1**2, 0.5)
        c_4 = c_2 + length * np.power(a_2**2 + b_2**2, 0.5)
        for point in points:
            distance = 0
            for abc in ((a_1, b_1, c_1), (a_1, b_1, c_3), (a_2, b_2, c_2), (a_2, b_2, c_4)):
                distance += get_distance_from_point_to_line(point, abc)
            distances.append(distance - 2 * length)
        return distances

    result = leastsq(_get_distances, [0, 1, 1, -center[1], -center[0]], maxfev=1000000, full_output=True)
    return result[0]


def change_angle(angle: float) -> float:
    """
    :param angle: angle in radian.
    :return: angle corresponding to given angle and lying in the range [0, 2 * pi].
    """

    if angle > 2 * np.pi:
        return change_angle(angle - 2 * np.pi)
    if angle < 0:
        return change_angle(angle + 2 * np.pi)
    return angle


def create_line(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    :param point_1: coordinates of first point;
    :param point_2: coordinates of second point.
    :return: coefficients (A, B, C) of a line passing through given points.
    """

    return point_2[1] - point_1[1], point_1[0] - point_2[0], point_2[0] * point_1[1] - point_1[0] * point_2[1]


def create_line_for_ray(x: float, y: float, angle: float) -> Tuple[float, float, float]:
    """
    :param x: coordinate X of point from which ray is drawn;
    :param y: coordinate Y of point from which ray is drawn;
    :param angle: angle to axis X beam tilt.
    :return: coefficients (A, B, C) of a line coinciding with given ray.
    """

    if angle != np.pi / 2 and angle != 3 * np.pi / 2:
        return -np.tan(angle), 1, x * np.tan(angle) - y
    return 1, 0, -x


def create_points_on_square(square: Square, center: Tuple[float, float], points_number: int, init_angle: float
                            ) -> List[np.ndarray]:
    """
    :param square: square on sides of which you need to create points;
    :param center: point from which rays will be drawn. Points of intersection of rays with sides of square will be the
    required points;
    :param points_number: number of rays that will be drawn from center point;
    :param init_angle: angle (in radians) to axis X at which the first ray will be drawn.
    :return: list with coordinates of points of intersection of sides of square with rays drawn from center point.
    """

    points = []
    for i in range(points_number):
        angle = np.pi * init_angle / 180 + 2 * np.pi / points_number * i
        point = square.find_intersection(*center, angle)
        if point is not None:
            points.append(point)
    return points


def find_intersection(line_1: Tuple[float, float, float], line_2: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """
    :param line_1: coefficients (A, B, C) of first line for writing in format A * x + B * y + C = 0;
    :param line_2: coefficients (A, B, C) of second line for writing in format A * x + B * y + C = 0.
    :return: coordinates of point of intersection of lines.
    """

    a = np.array([line_1[:2], line_2[:2]])
    b = np.array([-line_1[2], -line_2[2]])
    try:
        intersection = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        intersection = None
    return intersection


def get_distance_from_point_to_line(point: np.ndarray, line: Tuple[float, float, float]) -> float:
    """
    :param point: point coordinates;
    :param line: coefficients (A, B, C) of line for writing in format A * x + B * y + C = 0.
    :return: distance from point to line.
    """

    x, y = point
    a, b, c = line
    return np.abs(a * x + b * y + c) / np.power(a**2 + b**2, 0.5)


def get_distance_from_point_to_point(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
    """
    :param point_1: coordinates of first point;
    :param point_2: coordinates of second point.
    :return: distance between given points.
    """

    square_distance = 0
    for i in range(len(point_1)):
        square_distance += np.power(point_1[i] - point_2[i], 2)
    return np.power(square_distance, 0.5)


def plot_point(ax, point: Tuple[float, float], color: str, label: Optional[str] = None) -> None:
    """
    :param ax:
    :param point: coordinates of point to be drawn;
    :param color: color for point;
    :param label: label.
    """

    ax.scatter([point[0]], [point[1]], c=color, label=label)


def plot_points(ax, points: List[Tuple[float, float]], color: str) -> None:
    """
    :param ax:
    :param points: list with coordinates of points to be drawn;
    :param color: color for points.
    """

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    ax.scatter(x, y, c=color, label="Points on sides of square with errors")


def plot_sides(ax, side_coefficients: np.ndarray, length: float, color: str) -> None:
    """
    :param ax:
    :param side_coefficients: list with coefficients (A, B, C) of lines coinciding with sides of optimal square;
    :param length: square side length;
    :param color: color for sides of square found by least squares method.
    """

    a_1, a_2, b_1, c_1, c_2 = side_coefficients
    b_2 = -a_1 * a_2 / b_1
    c_3 = c_1 + length * np.power(a_1**2 + b_1**2, 0.5)
    c_4 = c_2 + length * np.power(a_2**2 + b_2**2, 0.5)

    corners = []
    coefficients = (a_1, b_1, c_1), (a_2, b_2, c_2), (a_1, b_1, c_3), (a_2, b_2, c_4)
    for i, line in enumerate(coefficients):
        corners.append(find_intersection(line, coefficients[i - 1]))

    for i, corner in enumerate(corners):
        x = np.linspace(corner[0], corners[i - 1][0], 100)
        y = np.linspace(corner[1], corners[i - 1][1], 100)
        if i == 0:
            ax.plot(x, y, "-.", c=color, label="Least squares method")
        else:
            ax.plot(x, y, "-.", c=color)
    calculated_center_x = np.mean([corner[0] for corner in corners])
    calculated_center_y = np.mean([corner[1] for corner in corners])
    plot_point(ax, (calculated_center_x, calculated_center_y), color, "Calculated center")


def plot_square(ax, square: Square, color: str) -> None:
    """
    :param ax:
    :param square: square to be drawn;
    :param color: color for square.
    """

    x = [point[0] for point in square.vertices]
    x.append(x[0])
    y = [point[1] for point in square.vertices]
    y.append(y[0])
    ax.set_aspect("equal")
    ax.plot(x, y, c=color, label="Square sides")
    plot_point(ax, square.center, color, "Square center")


def main() -> None:
    angle = 30
    length = 10
    x_square = 130
    y_square = 67
    square = Square(x_square, y_square, length, angle)

    center = 128, 65
    points_number = 40
    max_xy_shift = 0.5
    init_angle = np.pi / 2
    points = create_points_on_square(square, center, points_number, init_angle)
    points_with_error = add_error_to_points(points, max_xy_shift)
    side_coefficients = calculate_sides(points_with_error, length, center)

    fig, ax = plt.subplots()
    plot_square(ax, square, "red")
    plot_point(ax, center, "blue", "Approximate center of square")
    plot_points(ax, points_with_error, "green")
    plot_sides(ax, side_coefficients, length, "gray")
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    main()
