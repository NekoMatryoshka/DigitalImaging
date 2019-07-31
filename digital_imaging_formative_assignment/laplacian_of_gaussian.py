import numpy as np
import cv2
import laplacian
import gaussian


def boundary_padding(source_image, boundary_width):
    """
    boundary_padding(source_image, boundary_width) -> retval

    Return a padded image with given boundary width from given source image.

    :param source_image: the two-dimensional array form of the source image.
    :param boundary_width: the width of padding boundary.
    :return: a padded image of the source image.
    """
    inner_length = len(source_image)
    inner_width = len(source_image[0])

    outer_length = len(source_image) + boundary_width * 2
    outer_width = len(source_image[0]) + boundary_width * 2

    padded_image = np.zeros((outer_length, outer_width))

    # copying the center area
    for i in range(0, inner_length):
        for j in range(0, inner_width):
            padded_image[i + boundary_width][j + boundary_width] = source_image[i][j]

    # extrapolate values into the left, right, upper and down boundaries separately.
    for i in reversed(range(0, boundary_width)):
        for j in range(boundary_width, inner_width):
            padded_image[i][j] = padded_image[i + 1][j]

    for i in range(inner_length, outer_length):
        for j in range(boundary_width, inner_width):
            padded_image[i][j] = padded_image[i - 1][j]

    for i in range(0, outer_length):
        for j in reversed(range(0, boundary_width)):
            padded_image[i][j] = padded_image[i][j + 1]

    for i in range(0, outer_length):
        for j in range(inner_width, outer_width):
            padded_image[i][j] = padded_image[i][j - 1]

    return padded_image


def boundary_unpadding(padded_image, boundary_width):
    """
    boundary_unpadding(padded_image, boundary_width) -> retval

    Unpads a padded image and return it by given boundary width.

    :param padded_image: the two-dimensional array form of the padded image.
    :param boundary_width: the width of padding boundary.
    :return: a unpadded image of the given image.
    """
    inner_length = len(padded_image) - boundary_width * 2
    inner_width = len(padded_image[0]) - boundary_width * 2

    unpadded_image = np.zeros((inner_length, inner_width))

    for i in range(0, inner_length):
        for j in range(0, inner_width):
            unpadded_image[i][j] = padded_image[i + boundary_width][j + boundary_width]

    return unpadded_image


def filtering(source_image, gaussian_kernel_radius=3, gaussian_sigma=1.0):
    """
    filtering(source_image, gaussian_kernel_radius, gaussian_sigma) -> retval

    Returns a LoG filtered image by given source image and parameters.

    :param source_image: the unprocessed original image.
    :param gaussian_kernel_radius: the radius of the bilateral filter, the diameter of which is (radius * 2 + 1).
    :param gaussian_sigma: the sigma of gaussian function.
    :return: a LoG filtered image of the source image.
    """
    # pad the original by gaussian radius, blur it, and then unpad it.
    blurred_image = boundary_unpadding(gaussian.blurring(
        boundary_padding(source_image, gaussian_kernel_radius),
        gaussian_kernel_radius, gaussian_sigma), gaussian_kernel_radius)

    # get the edged image from the blurred image and unpad it.
    edged_image = boundary_unpadding(laplacian.edging(boundary_padding(blurred_image, 1)), 1)

    # sharpen the blurred image by blurred image and its edged image.
    sharpened_image = laplacian.sharpening(blurred_image, edged_image)

    return sharpened_image


# The main entry of this module. Run this entry to generate dLoG-filtered images.
if __name__ == '__main__':
    image = cv2.imread('source_image/face.png', cv2.IMREAD_GRAYSCALE)
    LoG_output_image = filtering(image)
    cv2.imwrite('filtered_image/face_LoG_output.png', LoG_output_image)
