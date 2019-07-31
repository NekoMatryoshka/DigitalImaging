import math
import numpy as np


def distance(x1, y1, x2, y2):
    """
    distance(x1, y1, x2, y2) -> int

    Returns the square of distance between point1 (x1, y1) and point2 (x2, y2).

    :param x1: the x-coordinate of point1.
    :param y1: the y-coordinate of point1.
    :param x2: the x-coordinate of point2.
    :param y2: the y-coordinate of point2.
    :return: the square of distance between point1 and point2.
    """
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def gaussian_function(x, sigma):
    """
    gaussian_function(x, sigma) -> float

    Returns the value of a gaussian function g(x) by given x and sigma.

    :param x: the variable of gaussian function g(x), usually proximity/similarity difference in bilateral filtering.
    :param sigma: the standard deviation constant, controlling the width of the bell curve.
    :return: the value of a gaussian function by given x and sigma.
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * (math.exp(-(x ** 2) / (2 * sigma ** 2)))


def __rendering(x, y, filtered_image, source_image, radius, sigma):
    """
    __rendering(x, y, filtered_image, source_image, radius, sigma) -> void

    A PRIVATE method that calculates the filtered intensity for a non-boundary point (x, y) by gaussian blurring.

    :param x: the x-coordinate of the filtered point.
    :param y: the y-coordinate of the filtered point.
    :param filtered_image: the two-dimensional array form of filtered image.
    :param source_image: the two-dimensional array form of the source image.
    :param radius: the radius of the bilateral filter, the diameter of which is (radius * 2 + 1).
    :param sigma: the sigma of gaussian function.
    :return: void
    """
    sum_of_filtered_intensity = 0.0
    sum_of_coefficient = 0.0

    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            gaussian = gaussian_function(distance(x, y, i, j), sigma)
            sum_of_filtered_intensity += gaussian * source_image[i][j]
            sum_of_coefficient += gaussian

    filtered_intensity = sum_of_filtered_intensity / sum_of_coefficient

    filtered_image[x][y] = int(round(filtered_intensity))


def blurring(source_image, radius, sigma):
    """
    blurring(source_image, radius, sigma) -> retval

    Returns a gaussian blurred image by given source image and parameters.

    :param source_image: the unprocessed original image.
    :param radius: the radius of the bilateral filter, the diameter of which is (radius * 2 + 1).
    :param sigma: the sigma of gaussian function.
    :return: a gaussian filtered image by given source image and parameters.
    """
    filtered_image = np.zeros(source_image.shape)

    for i in range(radius, len(source_image) - radius):
        for j in range(radius, len(source_image[0]) - radius):
            __rendering(i, j, filtered_image, source_image, radius, sigma)

    return filtered_image
