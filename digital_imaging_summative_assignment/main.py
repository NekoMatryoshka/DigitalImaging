"""
This module contains the methods of bilateral filters by different implementations,
    as well as a main method containing testing codes.

:author: Student Anonymous Marking Code Z0141503
:date: 06 Dec 2018
"""

import boundary
import bilateral
import util

import time

import numpy
import cv2


def grayscale_bilateral_filter(img, radius, sigma_proximity, sigma_intensity):
    """
    Bilateral-filters a grayscale image by boundary-padding, blurring and boundary-removing it.

    :param img: source image
    :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
    :param sigma_proximity: standard deviation of spatial proximity.
    :param sigma_intensity: standard deviation of intensity/colour similarity.
    :return: bilateral filtered image.
    """
    # use the intensity_distance function for single colour channel as intensity distance function
    return boundary.removal(
        bilateral.filter(boundary.replication(img, radius),
                         radius, sigma_proximity, sigma_intensity, util.intensity_distance), radius)


def rgb_bilateral_filter_by_combing_each_channel(img, radius, sigma_proximity, sigma_intensity):
    """
    Bilateral-filters a RGB image by spliting it to three channels, filtering each of them separately
        and then stacking them back.

    :param img: source image
    :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
    :param sigma_proximity: standard deviation of spatial proximity.
    :param sigma_intensity: standard deviation of intensity/colour similarity.
    :return: bilateral filtered image.
    """
    b, g, r = cv2.split(img)

    b = grayscale_bilateral_filter(b, radius, sigma_proximity, sigma_intensity)
    g = grayscale_bilateral_filter(g, radius, sigma_proximity, sigma_intensity)
    r = grayscale_bilateral_filter(r, radius, sigma_proximity, sigma_intensity)

    return numpy.dstack((b, g, r))


def rgb_bilateral_filter_by_integrating(img, radius, sigma_proximity, sigma_intensity):
    """
    Bilateral-filters a RGB image by filtering each pixel as a vector based on its RGB Euclidean colour similarity.

    :param img: source image
    :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
    :param sigma_proximity: standard deviation of spatial proximity.
    :param sigma_intensity: standard deviation of intensity/colour similarity.
    :return: bilateral filtered image.
    """
    # use the rgb_distance function for Euclidean colour distance as intensity distance function
    return boundary.removal(
        bilateral.filter(boundary.replication(img, radius),
                         radius, sigma_proximity, sigma_intensity, util.rgb_distance), radius)


def lab_bilateral_filter(img, radius, sigma_proximity, sigma_intensity):
    """
    Bilateral-filters a RGB image by converting into L*ab colour space, filtering it, and converting back to RGB image.

    :param img: source image
    :param radius: radius of the bilateral mask, the diameter of which is (radius * 2 + 1).
    :param sigma_proximity: standard deviation of spatial proximity.
    :param sigma_intensity: standard deviation of intensity/colour similarity.
    :return: bilateral filtered image.
    """
    # pad boundary.
    res = boundary.replication(img, radius)

    # for each RGB pixel, convert to L*ab colour vectors.
    for i in range(0, len(res)):
        for j in range(0, len(res[0])):
            res[i][j] = util.rgb2lab(res[i][j])

    # use the rgb_distance function for perceptual colour distance as intensity distance function
    res = bilateral.filter(res, radius, sigma_proximity, sigma_intensity, util.lab_distance)

    # for each filtered L*ab pixel, convert back to RGB vectors.
    for i in range(0, len(res)):
        for j in range(0, len(res[0])):
            res[i][j] = util.lab2rgb(res[i][j])

    # remove boundary.
    return boundary.removal(res, radius)


def main():
    """
    Main function contains testing code of above functions with selected parameters.

    :return: void.
    """
    # read in grayscale and colour images.
    image_a = cv2.imread('input\imageA.png', cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread("input\imageB.png", cv2.IMREAD_COLOR)

    # filter grayscale image by a set of combination of sigma parameters.
    for p in [1, 15, 1000]:
        for i in [1, 15, 1000]:
            output = grayscale_bilateral_filter(image_a, 5, p, i)
            cv2.imwrite('output\output_a_p' + str(p) + '_i' + str(i) + '.png', output)
            print('output_a_p' + str(p) + '_i' + str(i))

    # filter colour image by different approaches and sigma parameters.
    output = rgb_bilateral_filter_by_combing_each_channel(image_b, 5, 15, 15)
    cv2.imwrite('output\output_b_rgb_combing_pi15.png', output)
    print('output_b_rgb_combing_pi15')

    output = rgb_bilateral_filter_by_integrating(image_b, 5, 15, 15)
    cv2.imwrite('output\output_b_rgb_integrating_pi15.png', output)
    print('output_b_rgb_integrating_pi15')

    output = lab_bilateral_filter(image_b, 5, 15, 15)
    cv2.imwrite('output\output_b_lab_pi15.png', output)
    print('output_b_lab_pi15')

    output = rgb_bilateral_filter_by_combing_each_channel(image_b, 5, 1000, 1000)
    cv2.imwrite('output\output_b_rgb_combing_pi1000.png', output)
    print('output_b_rgb_combing_pi1000')

    output = rgb_bilateral_filter_by_integrating(image_b, 5, 1000, 1000)
    cv2.imwrite('output\output_b_rgb_integrating_pi1000.png', output)
    print('output_b_rgb_integrating_pi1000')

    output = lab_bilateral_filter(image_b, 5, 1000, 1000)
    cv2.imwrite('output\output_b_lab_pi1000.png', output)
    print('output_b_lab_pi1000')


# main entrance of the program.
if __name__ == '__main__':
    print('START!')
    start = time.time()
    main()
    end = time.time()
    print('ALL DONE! total running time is ' + str(end - start))
