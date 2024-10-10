import os

import cv2 as cv
import numpy as np

# Ref Doc: https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html

"""
UTILITY FUNCTIONS
"""


def make_grid(caption, img1, img2, img3, img4):
    vertical_stacked_resized_images_1 = np.vstack((img1, img2))
    vertical_stacked_resized_images_2 = np.vstack((img3, img4))
    horizontal_resized_stacked_1 = np.hstack(
        (vertical_stacked_resized_images_1, vertical_stacked_resized_images_2)
    )
    cv.imshow(caption, horizontal_resized_stacked_1)
    return horizontal_resized_stacked_1


def add_image_name(image, image_name):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2
    image_name = image_name.split(".")[0]
    # Get the size of the text
    (text_width, text_height), _ = cv.getTextSize(
        image_name, font, font_scale, thickness
    )

    # Calculate the position for right-aligned text
    margin = 20  # Margin from the right and bottom edges
    position = (
        image.shape[1] - text_width - margin,  # X coordinate
        image.shape[0] - margin,  # Y coordinate
    )

    # Ensure the text fits within the image bounds
    if position[0] < 0:
        position = (0, position[1])
    if position[1] < text_height:
        position = (position[0], text_height)

    # Add text to the image
    cv.putText(
        image,
        image_name,
        position,
        font,
        font_scale,
        font_color,
        thickness,
        cv.LINE_AA,
    )

    return image


def resize_images_to_same_size(images, size=(320, 400)):
    return [cv.resize(img, size) for img in images]


"""
BLUR FUNCTIONS
"""


def applyMedianBlur(base_image, kernel_size, iterations):
    new_image = base_image
    for _ in range(iterations):
        # Median Blur: replaces each pixel value with the median value of the neighborhood pixels
        # Helps in reducing noise while preserving edges
        new_image = cv.medianBlur(new_image, kernel_size)
    return new_image


def applyBilateralFilter(base_image, diameter, sigma_colour, sigma_space, iterations):
    bilateral_image = base_image
    for _ in range(iterations):
        # Bilateral Filter: performs convolution while considering both spatial distance and color intensity
        # Helps in preserving edges while smoothing the image, making it suitable for cartoon-like effects
        bilateral_image = cv.bilateralFilter(
            bilateral_image, diameter, sigma_colour, sigma_space
        )
    return bilateral_image


"""
TRANSFORMATION FUNCTIONS
"""


def make_watercolored_image(image):
    # STEP 1: Resize and Preprocess the image
    image_dimensions = (320, 400)
    image = cv.resize(image, image_dimensions)

    # Step 2: Removing impurities from image and apply Smoothing Using Edge-Preserving Filter
    clear_image = applyMedianBlur(base_image=image, kernel_size=3, iterations=1)
    image_edge_preserve_filter = cv.edgePreservingFilter(
        clear_image, sigma_s=45, sigma_r=0.3
    )

    # Step 3: Combine the Effects to Create a Watercolor Look
    bilateral_filter_image = applyBilateralFilter(
        image_edge_preserve_filter,
        diameter=7,
        sigma_colour=20,
        sigma_space=20,
        iterations=2,
    )

    bilateral_filter_image = applyBilateralFilter(
        bilateral_filter_image,
        diameter=5,
        sigma_colour=30,
        sigma_space=30,
        iterations=1,
    )

    # Step 4: Tune the Art
    kernel_size = (3, 3)
    standard_deviation_in_x_y = 0.8
    gaussian_mask = cv.GaussianBlur(
        bilateral_filter_image, kernel_size, standard_deviation_in_x_y
    )

    return gaussian_mask


"""
MAIN FUNCTION
"""

image_directory = "images/"
image_names = sorted(os.listdir(image_directory))[:4]

original_images = [
    cv.imread(os.path.join(image_directory, name)) for name in image_names
]
watercolored_images = [make_watercolored_image(img) for img in original_images]

original_resized = resize_images_to_same_size(original_images)
watercolored_resized = resize_images_to_same_size(watercolored_images)

original_with_names = [
    add_image_name(original_resized[i], image_names[i]) for i in range(4)
]
watercolored_with_names = [
    add_image_name(watercolored_resized[i], image_names[i]) for i in range(4)
]

# Step 5: Display and Save the Result:
original_grid = make_grid("Original Images", *original_with_names)
watercolored_grid = make_grid("Watercolored Images", *watercolored_with_names)

if not os.path.exists("output_images/"):
    os.mkdir("output_images")

cv.imwrite("output_images/original.png", original_grid)
cv.imwrite("output_images/Watercolored.png", watercolored_grid)
print("Successfully saved images in 'output_images' directory")

cv.waitKey(0)
cv.destroyAllWindows()
