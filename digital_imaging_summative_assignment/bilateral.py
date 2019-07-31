"""
This module contains the implementation of bilateral filtering algorithm.

:author: Student Anonymous Marking Code Z0141503
:date: 06 Dec 2018
"""

import util

import numpy


def filter(img, radius, sigma_proximity, sigma_intensity, intensity_distance_function):
    """
    Returns a bilateral blurred image. The function only filters the non-boundary area of the source image.

    :param img: source image.
    :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
    :param sigma_proximity: standard deviation of spatial proximity.
    :param sigma_intensity: standard deviation of intensity/colour similarity.
    :param intensity_distance_function: the function that calculates the intensity/colour distance of two pixels.
    :return: bilateral blurred image
    """

    def __mask(x, y, res, img, radius, sigma_proximity, sigma_intensity, intensity_distance_function):
        """
        Calculates the filtered intensity for a point (x, y) by bilateral filtering mask.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param res: a pointer/reference of the filtered image.
        :param img: source image
        :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
        :param sigma_proximity: standard deviation of spatial proximity.
        :param sigma_intensity: standard deviation of intensity/colour similarity.
        :param intensity_distance_function: the function that calculates the intensity/colour distance of two pixels.
        :return: void
        """
        # the numerator sum(g(|p - p'|) * g(|I-I'|) * I').
        sum_i = 0.0
        # the denominator sum(g(|p - p'|) * g(|I-I'|)).
        sum_p = 0.0

        # for each pixel around point (x, y) that is inside radius.
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                # calculate intensity gaussian function g(|I-I'|).
                gaussian_intensity = util.gaussian(intensity_distance_function(img[x][y], img[i][j]), sigma_intensity)
                # calculate proximity gaussian function g(|p - p'|).
                gaussian_proximity = util.gaussian(util.spatial_distance(x, y, i, j), sigma_proximity)
                # calculate the product of two gaussian function.
                product = gaussian_intensity * gaussian_proximity

                # add up to the numerator and denominator of bilateral filtering algorithm to calculate the sums.
                sum_i += product * img[i][j]
                sum_p += product

        # calculate the filtered intensity of point (x, y)
        res[x][y] = sum_i / sum_p

    res = numpy.zeros(img.shape)

    # for each pixel in center area, calculate its filtered intensity.
    for i in range(radius, len(img) - radius):
        for j in range(radius, len(img[0]) - radius):
            __mask(i, j, res, img, radius, sigma_proximity, sigma_intensity, intensity_distance_function)

    return res
