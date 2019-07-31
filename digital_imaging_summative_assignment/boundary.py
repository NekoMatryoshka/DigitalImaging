"""
A utility model that contains methods for boundary padding by replication.

@author: Student Anonymous Marking Code Z0141503
@date: 06 Jan 2018
"""


import numpy


def replication(img, width):
    """
    Pads an image by given boundary width.

    :param img: source image.
    :param width: width of padding boundary.
    :return: padded image.
    """
    inner_length = len(img)
    inner_width = len(img[0])

    outer_length = len(img) + width * 2
    outer_width = len(img[0]) + width * 2

    # determine the dimension of return array by source image.
    if len(img.shape) == 2:
        res = numpy.zeros((outer_length, outer_width))
    else:
        res = numpy.zeros((outer_length, outer_width, 3))

    # copying the center area
    for i in range(0, inner_length):
        for j in range(0, inner_width):
            res[i + width][j + width] = img[i][j]

    # extrapolate values into the left, right, upper and down boundaries separately.
    for i in reversed(range(0, width)):
        for j in range(width, inner_width):
            res[i][j] = res[i + 1][j]

    for i in range(inner_length, outer_length):
        for j in range(width, inner_width):
            res[i][j] = res[i - 1][j]

    for i in range(0, outer_length):
        for j in reversed(range(0, width)):
            res[i][j] = res[i][j + 1]

    for i in range(0, outer_length):
        for j in range(inner_width, outer_width):
            res[i][j] = res[i][j - 1]

    return res


def removal(img, width):
    """
    Removes the boundary of a image by given boundary width.

    :param img: source image
    :param width: width of removed boundary.
    :return: processed image with boundary removed.
    """
    inner_length = len(img) - width * 2
    inner_width = len(img[0]) - width * 2

    if len(img.shape) == 2:
        res = numpy.zeros((inner_length, inner_width))
    else:
        res = numpy.zeros((inner_length, inner_width, 3))

    for i in range(0, inner_length):
        for j in range(0, inner_width):
            res[i][j] = img[i + width][j + width]

    return res
