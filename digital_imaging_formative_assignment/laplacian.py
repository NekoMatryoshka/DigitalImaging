import numpy as np
import cv2


def __rendering(x, y, filtered_image, source_image):
    """
    __rendering(x, y, filtered_image, source_image) -> void

    A PRIVATE method that calculates the filtered intensity for a non-boundary point (x, y) by laplacian operator.
    The method implements a fixed 3*3 4-neighbourhood mask.

    :param x: the x-coordinate of the filtered point.
    :param y: the y-coordinate of the filtered point.
    :param filtered_image: the two-dimensional array form of filtered image.
    :param source_image: the two-dimensional array form of the source image.
    :return: void
    """
    filtered_image[x][y] = -4 * source_image[x][y] + source_image[x - 1][y] + source_image[x + 1][y] + source_image[x][
        y - 1] + source_image[x][y + 1]


def edging(source_image):
    """
    edging(source_image) -> retval

    Returns a laplacian edged image by given source image and parameters.

    :param source_image: the two-dimensional array form of the source image.
    :return: a laplacian filtered image by given source image and parameters.
    """
    filtered_image = np.zeros(source_image.shape)

    for i in range(1, len(source_image) - 1):
        for j in range(1, len(source_image[0]) - 1):
            __rendering(i, j, filtered_image, source_image)

    return filtered_image


def sharpening(source_image, filtered_image):
    """
    sharpening(source_image, filtered_image) -> retval

    Returns a laplacian sharpened image by subtracting a source image by a filtered image.

    :param source_image: the two-dimensional array form of the source image.
    :param filtered_image: the two-dimensional array form of a laplacian edged image from the source image.
    :return: a laplacian sharpened image of the source image.
    """
    return np.array(source_image) - np.array(filtered_image)


# The main entry of this module. Run this entry to generate directly laplacian-filtered images.
if __name__ == '__main__':
    image = cv2.imread('source_image/face.png', cv2.IMREAD_GRAYSCALE)

    edged_image = edging(image)
    cv2.imwrite('filtered_image/face_directly_edged_output.png', edged_image)

    sharpened_image = sharpening(image, edged_image)
    cv2.imwrite('filtered_image/face_directly_sharpened_output.png', sharpened_image)
