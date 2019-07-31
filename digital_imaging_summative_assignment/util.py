"""
A utility model that abstracts math operations related to intensity and spatial distance, and gaussian function.

@author: Student Anonymous Marking Code Z0141503
@date: 06 Dec 2018
"""


import math
import numpy


def gaussian(x, sigma):
    """
    Returns the value of a gaussian function.

    :param x: the variable of gaussian function g(x), usually the spatial/intensity distance of two pixels.
    :param sigma: the standard deviation constant, controlling the width of the bell curve.
    :return: the value of a gaussian function by given x and sigma.
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * (math.exp(-(x ** 2) / (2 * sigma ** 2)))


def rgb2lab(rgb):
    """
    Converts the intensity vector of a pixel from RGB to L*ab colour space.

    :param rgb: RGB color vector of the pixel, in the form of [B, G, R].
    :return: the corresponding L*ab vector, in the form of [L, a, b].
    """
    # normalization
    b = rgb[0] / 255.0
    g = rgb[1] / 255.0
    r = rgb[2] / 255.0

    # gamma mapping
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92

    # convert RGB to XYZ colour space and normalization
    x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.95047
    y = (0.212671 * r + 0.715160 * g + 0.072169 * b) / 1.0
    z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883

    # XYZ spatial mapping
    fx = x ** (1.0 / 3.0) if x > 0.008856 else 7.787 * x + 0.137931
    fy = y ** (1.0 / 3.0) if y > 0.008856 else 7.787 * y + 0.137931
    fz = z ** (1.0 / 3.0) if z > 0.008856 else 7.787 * z + 0.137931

    # convert XYZ to Lab colour space
    l = 116 * fy - 16 if y > 0.008856 else 903.3 * y
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return numpy.array((l, a, b))


def lab2rgb(lab):
    """
    Converts the intensity vector of a pixel from L*ab to RGB colour space.

    :param lab: L*ab color vector of the pixel, in the form of [L, a, b].
    :return: the corresponding RGB vector, in the form of [B, G, R].
    """
    y = (lab[0] + 16) / 116
    x = lab[1] / 500 + y
    z = y - lab[2] / 200

    x = 0.95047 * (x ** 3 if x ** 3 > 0.008856 else (x - 0.137931) / 7.787)
    y = 1.0 * (y ** 3 if y ** 3 > 0.008856 else (y - 0.137931) / 7.787)
    z = 1.08883 * (z ** 3 if z ** 3 > 0.008856 else (z - 0.137931) / 7.787)

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    r = 1.055 * r ** 0.416667 - 0.055 if r > 0.0031308 else 12.92 * r
    g = 1.055 * g ** 0.416667 - 0.055 if g > 0.0031308 else 12.92 * g
    b = 1.055 * b ** 0.416667 - 0.055 if b > 0.0031308 else 12.92 * b

    return numpy.array((max(0, min(1, b)) * 255,
                        max(0, min(1, g)) * 255,
                        max(0, min(1, r)) * 255))


def rgb_distance(rgb1, rgb2):
    """
    Returns the Euclidean colour distance between two RGB vectors.

    :param rgb1: RGB color vector of the pixel1.
    :param rgb2: RGB color vector of the pixel2.
    :return: the Euclidean distance between two RGB vectors
    """
    return math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)


def lab_distance(lab1, lab2):
    """
    Returns the perceptual colour distance between two L*ab vectors by CIE76 formula.

    :param lab1: L*ab color vector of the pixel1.
    :param lab2: L*ab color vector of the pixel2.
    :return: the perceptual colour distance between two RGB vectors
    """
    delta_l = lab1[0] - lab2[0]
    delta_a = lab1[1] - lab2[1]
    delta_b = lab1[2] - lab2[2]

    c1 = math.sqrt(lab1[1] ** 2 + lab1[2] ** 2)
    c2 = math.sqrt(lab2[1] ** 2 + lab2[2] ** 2)

    delta_c = c1 - c2

    delta_h = delta_a ** 2 + delta_b ** 2 - delta_c ** 2
    delta_h = math.sqrt(delta_h) if delta_h > 0 else 0

    delta_e = math.sqrt(delta_l ** 2 + (delta_c / (1.0 + 0.045 * c1)) ** 2 + (delta_h / (1.0 + 0.015 * c1)) ** 2)

    return delta_e


def spatial_distance(x1, y1, x2, y2):
    """
    Returns the spatial distance between pixel1 (x1, y1) and pixel2 (x2, y2).

    :param x1: the x-coordinate of point1.
    :param y1: the y-coordinate of point1.
    :param x2: the x-coordinate of point2.
    :param y2: the y-coordinate of point2.
    :return: the spatial distance between point1 and point2.
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def intensity_distance(intensity1, intensity2):
    """
    Returns the absolute value of difference between the intensity of two pixels.

    :param intensity1: the intensity of pixels.
    :param intensity2: the intensity of pixels.
    :return: the absolute difference of intensity of two points.
    """
    return abs(int(intensity1) - int(intensity2))
