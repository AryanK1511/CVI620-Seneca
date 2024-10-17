import cv2 as cv
import numpy as np

images = {
    "Geometric Shapes - OpenCV": "images/geometric_shapes.jpg",
    "Industrial Objects - OpenCV": "images/industrial_objects.jpg",
    "Natural Scenes - OpenCV": "images/natural_scenes.jpg",
}

# Task 1: Acquiring Sample Images
for window_name, image_path in images.items():
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        continue

    cv.imshow(window_name, image)

    # Task 2: Grayscale Conversion
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow(f"Grayscale - {window_name}", gray_image)

    # Task 3: Noise Reduction Using Smoothing
    gaussian_blur = cv.GaussianBlur(gray_image, (5, 5), 0)
    gray_image_3channel = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
    gaussian_blur_3channel = cv.cvtColor(gaussian_blur, cv.COLOR_GRAY2BGR)
    concatenated_image = cv.hconcat(
        [image, gray_image_3channel, gaussian_blur_3channel]
    )
    cv.imshow(
        f"Original vs Grayscale vs Gaussian Blurred - {window_name}", concatenated_image
    )

    # Task 4: Sobel Edge Detection
    sobel_x = cv.Sobel(gaussian_blur, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(gaussian_blur, cv.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)
    sobel_combined = np.uint8(sobel_combined)
    cv.imshow(f"Sobel Edge Detection - {window_name}", sobel_combined)

    # Task 5: Laplacian Edge Detection
    laplacian = cv.Laplacian(gaussian_blur, cv.CV_64F)
    laplacian = np.abs(laplacian)
    laplacian = np.uint8(laplacian)
    cv.imshow(f"Laplacian Edge Detection - {window_name}", laplacian)

    # Task 7: Canny Edge Detection
    canny_edges = cv.Canny(gaussian_blur, 100, 200)
    cv.imshow(f"Canny Edge Detection - {window_name}", canny_edges)

cv.waitKey(0)
cv.destroyAllWindows()
