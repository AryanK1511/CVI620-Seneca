# CVI620 Midterm Exam - Final Report

- **Name:** Aryan Khurana
- **Student ID:** 145282216

## Introduction

Edge detection is a fundamental technique in computer vision that identifies boundaries within an image where significant changes in pixel intensity occur. These boundaries often represent important features such as object contours, texture changes, or depth discontinuities.

Edge detection is crucial in various applications including:

- Object detection and recognition
- Image segmentation
- Feature extraction
- Medical image processing
- Industrial quality control
- Autonomous vehicle navigation

In this project, I will be walking you through three edge detection techniques, **Sobel**, **Laplacian**, and **Canny**.

## Methods and Step by Step Process

This project is divided into 3 major steps:

1. **Image Acquisition and Preprocessing**
2. **Edge Detection Techniques**
3. **Performance Analysis and Optimization**

Let us go through all of them and understand edge detection in detail.

## Part A: Image Acquisition and Preprocessing

### Task 1: Acquiring Sample Images

- Lets start off by loading and displaying a series of images from specified file paths stored in the `image_hashmap` dictionary.
- Each key in the dictionary represents a window name for displaying the corresponding image. T
- The code iterates through the items in the dictionary, reading each image using OpenCV's `cv.imread()` function. Successfully loaded images are appended to the `images` list, and each image is displayed in a separate window with its designated name using `cv.imshow()`.

```python
import cv2 as cv  # Import the OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations (not used in this snippet)

# Dictionary to map window names to their corresponding image file paths
image_hashmap = {
    "Geometric Shapes - OpenCV": "images/geometric_shapes.jpg",
    "Industrial Objects - OpenCV": "images/industrial_objects.jpg",
    "Natural Scenes - OpenCV": "images/natural_scenes.jpg",
}

images = []  # List to store successfully loaded images

# Iterate through the dictionary to load and display each image
for window_name, image_path in image_hashmap.items():
    # Read the image from the specified file path
    image = cv.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")  # Print an error message if loading fails
        continue  # Skip to the next iteration if the image is not loaded

    images.append(image)  # Add the successfully loaded image to the list
    cv.imshow(window_name, image)  # Display the image in a window with the specified name

# Wait indefinitely for a key press
cv.waitKey(0)

# Close all open image windows
cv.destroyAllWindows()
```

![img](https://drive.google.com/thumbnail?id=1fP2hQUyfJz-37vOPlPnY2wsTtibosb98&sz=w1000)

### Task 2: Grayscale Conversion

Next, we will convert all our images to grayscale and will store them in the `gray_images` list.

```python
gray_images = []  # List to store the grayscale versions of the images

# Iterate through each image in the 'images' list (loaded from the previous code)
for image in images:
    # Convert the current image from BGR (color) to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Append the grayscale image to the 'gray_images' list
    gray_images.append(gray_image)
```

**Grayscale conversion simplifies edge detection by:**

- Reducing image complexity to a single intensity channel
- Focusing on intensity changes rather than color variations
- Improving computational efficiency
- Reducing noise in the edge detection process

![img](https://drive.google.com/thumbnail?id=1kVmdJIru_c60Z3CJHxsgehDPL1cseBlP&sz=w1000)

### Task 3: Noise Reduction Using Smoothing

- Smoothing an image, or blurring, helps reduce noise from sensor limitations or poor lighting, resulting in a cleaner representation of the scene.
- By minimizing unwanted variations, smoothing allows algorithms to focus on significant features while enhancing edge detection, making it easier to identify important transitions in intensity.
- It also improves image segmentation by clarifying distinctions between regions and enhances visual appeal by softening harsh edges.

```python
blurred_images = []  # List to store the Gaussian blurred versions of the grayscale images

# Iterate through each grayscale image in the 'gray_images' list
for i in range(len(gray_images)):
    # Apply Gaussian blur to the current grayscale image with a kernel size of (5, 5)
    gaussian_blur = cv.GaussianBlur(gray_images[i], (5, 5), 0)
    blurred_images.append(gaussian_blur)  # Store the blurred image in the list

    # Convert the current grayscale image back to a 3-channel BGR image
    gray_image_3channel = cv.cvtColor(gray_images[i], cv.COLOR_GRAY2BGR)

    # Convert the blurred image to a 3-channel BGR image
    gaussian_blur_3channel = cv.cvtColor(gaussian_blur, cv.COLOR_GRAY2BGR)

    # Concatenate the original image, grayscale image, and Gaussian blurred image horizontally
    concatenated_image = cv.hconcat(
        [images[i], gray_image_3channel, gaussian_blur_3channel]
    )

    # Display the concatenated image in a window with a descriptive title
    cv.imshow(
        f"Original vs Grayscale vs Gaussian Blurred - Image {i + 1}",
        concatenated_image
    )

# Wait indefinitely for a key press
cv.waitKey(0)

# Close all open image windows
cv.destroyAllWindows()
```

**Gaussian blur** smooths the image to reduce noise while preserving edge information, using a 5x5 kernel size.

Here is a comparison between the original image, grayscale image and image after applying gaussian blur.

![img](https://drive.google.com/thumbnail?id=1lzZf-7f7EHpkKC_-fFhPOFvVWZGJDS4f&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1PuoQyFcQlJxbB_uDycQucZUbquvNZZIk&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1NiQwO28NJfaF3l0Ws6JQGgp0UvpBKAxW&sz=w1000)

## Part A: Edge Detection Techniques

### Task 4: Sobel Edge Detection

- The primary goal of the Sobel operator is to find edges in an image. Edges are the boundaries where there are significant changes in brightness or color.
- The Sobel operator uses two filters (or kernels) that move across the image. One filter detects horizontal edges (left to right), while the other detects vertical edges (top to bottom). As these filters are applied to the image, they calculate how much the pixel values change in both directions.
- After applying both filters, the Sobel operator combines the results to determine the overall edge strength at each pixel. This helps highlight areas where there are significant transitions in color or brightness.
- The output of the Sobel operator is an image where the edges are more prominent. Typically, brighter areas in the output represent strong edges, while darker areas indicate little or no edge presence.

```python
# Define a list of kernel sizes to use for the Sobel operator
kernel_sizes = [3, 5, 7]
sobel_images = []  # List to store the Sobel edge-detected images

# Iterate through each blurred image in the 'blurred_images' list
for blurred_image in blurred_images:
    sobel_images = []  # Reset the list for each blurred image

    # Iterate through each kernel size defined in 'kernel_sizes'
    for ksize in kernel_sizes:
        # Apply the Sobel operator in the x direction to detect horizontal edges
        sobel_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=ksize)

        # Apply the Sobel operator in the y direction to detect vertical edges
        sobel_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=ksize)

        # Combine the absolute values of the Sobel x and y results to create a single edge image
        sobel_combined = cv.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)

        # Convert the combined image to an unsigned 8-bit integer format
        sobel_combined = np.uint8(sobel_combined)

        # Create a window name for displaying the Sobel edge detection results
        window_name = f"Sobel Edge Detection - Kernel Size: {ksize}"

        # Display the Sobel edge-detected image in a window
        cv.imshow(window_name, sobel_combined)

    # Append the last Sobel combined image for the current blurred image to the list
    sobel_images.append(sobel_combined)

cv.waitKey(0)
cv.destroyAllWindows()
```

#### Results of Experimenting with Different Kernel Sizes

When you experiment with different kernel sizes (3, 5, and 7) using the Sobel operator, you can observe different effects:

1. **Smaller Kernel Sizes (e.g., 3)**:

   - Smaller kernels capture fine details and edges more precisely.
   - The output image shows sharper edges, but it may also highlight noise, making some unwanted details more visible.

2. **Medium Kernel Sizes (e.g., 5)**:

   - This size offers a balance between detail and noise reduction.
   - The edges are well-defined, and the image looks cleaner, as some noise is smoothed out without losing too much detail.

3. **Larger Kernel Sizes (e.g., 7)**:

   - Larger kernels smooth the image more significantly, reducing noise further.
   - While the important edges are still visible, finer details may be lost. The output appears less noisy but may miss some smaller edges. In the example below, you will see how we don't even see edges properly anymore when we go for a `7x7` Kernel size.

#### Comparison (`3x3` vs `5x5` vs `7x7`)

![img](https://drive.google.com/thumbnail?id=1e81l2pkdexlMAbWmTmXUkQf-yIIRxQoQ&sz=w1000)

By changing the kernel size in the Sobel operator, you can influence how edges are detected in an image. Smaller kernels provide more detail, while larger kernels offer a smoother appearance, which can be beneficial in various image processing applications depending on the desired outcome.

### Task 5: Laplacian Edge Detection

The **Laplacian edge detection** technique is a method used to find edges in images. It works by looking for places where there’s a sudden change in color or brightness.

- The Laplacian operator scans through the image and looks for areas where the colors change quickly. For example, where a bright object meets a dark background, there’s a clear edge.
- When the Laplacian operator identifies these sudden changes, it emphasizes them. The result is a new image where edges are shown very clearly, making it easier to see the boundaries of different objects in the picture.
- The new image created by the Laplacian operator shows bright areas where the edges are, while the rest of the image appears darker. This contrast helps to visualize the shapes and outlines within the image effectively.

```python
laplacian_images = []  # Initialize a list to store images with Laplacian edge detection.

# Loop through each blurred image and its index.
for index, blurred_image in enumerate(blurred_images):
    # Apply the Laplacian operator to detect edges in the blurred image.
    laplacian = cv.Laplacian(blurred_image, cv.CV_64F)

    # Take the absolute value of the Laplacian result to ensure all edge values are positive.
    laplacian = np.abs(laplacian)

    # Convert the Laplacian result back to an 8-bit unsigned integer format for displaying.
    laplacian = np.uint8(laplacian)

    # Append the processed Laplacian image to the list.
    laplacian_images.append(laplacian)

    # Display the Laplacian edge-detected image in a window.
    cv.imshow(f"Laplacian Edge Detection - Image {index + 1}", laplacian)

cv.waitKey(0)
cv.destroyAllWindows()
```

![img](https://drive.google.com/thumbnail?id=1Y4touS8oi-P3B3aR8SUwo9BUN9fElhtQ&sz=w1000)

#### Differences from Gradient-Based Methods (Like Sobel)

1. **Focus on Changes**:

   - The **Sobel operator** focuses on how colors change in a specific direction, like left to right or up and down. It helps in understanding where lines or edges are located.
   - The **Laplacian operator**, on the other hand, looks for any rapid change in color, regardless of direction. It detects edges from all angles, giving a more general view of where the edges are.

2. **Handling Noise**:

   - The Sobel operator is generally better at ignoring small disturbances or noise in the image, which helps in producing cleaner edges.
   - The Laplacian can be more sensitive to noise. It might highlight not just the edges but also random changes, which can lead to some unnecessary details appearing in the output.

3. **Type of Edges Detected**:

   - The Sobel operator shows lines and boundaries more clearly by emphasizing the direction of edges, making it ideal for detecting straight lines.
   - The Laplacian can show all significant edges, including curves and corners, making it useful for more complex shapes.

4. **Combined Use**:
   - Often, both methods are used together. The Sobel operator can first identify where the main edges are, and then the Laplacian can refine those edges by emphasizing additional details.

#### Sobel (`7x7` Kernel Size) v/s Laplacian

![img](https://drive.google.com/thumbnail?id=1C6AEooeyCUDJafQio1amksgrFqed5C7p&sz=w1000)

#### c) Canny Edge Detection

```python
thresholds = [(50, 150), (100, 200), (150, 250)]
for low_thresh, high_thresh in thresholds:
    canny_edges = cv.Canny(blurred_image, low_thresh, high_thresh)
```

The Canny algorithm:

- Uses double thresholding to detect strong and weak edges
- Implements hysteresis to connect edge segments
- Provides more precise edge detection compared to Sobel and Laplacian

## Results

### Comparative Analysis

1. **Sobel Edge Detection**

   - Effectively detected strong edges in different directions
   - Performance varied with kernel size:
     - 3x3: Captured fine details but more sensitive to noise
     - 5x5: Balanced detail and noise reduction
     - 7x7: Smoother edges but lost some fine details

2. **Laplacian Edge Detection**

   - Detected edges regardless of orientation
   - More sensitive to noise compared to Sobel
   - Produced thinner edges but with more false positives

3. **Canny Edge Detection**
   - Provided the most complete and clean edge maps
   - Different threshold combinations affected results:
     - (50, 150): Detected more edges but included some noise
     - (100, 200): Balanced edge detection
     - (150, 250): Focused on strongest edges only

## Challenges

1. **Noise Handling**

   - Challenge: Initial edge detection attempts produced noisy results
   - Solution: Implemented Gaussian blur preprocessing with a 5x5 kernel
   - Impact: Significantly reduced false edge detection while preserving important features

2. **Parameter Tuning**

   - Challenge: Different images required different parameters for optimal results
   - Solution: Implemented multiple parameter combinations (kernel sizes, thresholds)
   - Impact: Allowed for comparison and selection of best parameters for each image type

3. **Image Size Consistency**
   - Challenge: Comparison of different methods required consistent image sizes
   - Solution: Implemented resize operations when concatenating images
   - Impact: Enabled direct visual comparison of different edge detection methods

## Conclusion

### Key Findings

1. Canny edge detection provided the most reliable results across different image types
2. Preprocessing steps (grayscale conversion and Gaussian blur) were crucial for good results
3. Parameter selection significantly impacts edge detection quality

### Future Improvements

1. Implement adaptive parameter selection based on image characteristics
2. Explore deep learning-based edge detection methods
3. Develop automated evaluation metrics for edge detection quality
4. Investigate multi-scale edge detection approaches

### Recommendations

1. Use Canny edge detection for general-purpose edge detection tasks
2. Apply appropriate preprocessing steps before edge detection
3. Consider the specific requirements of the application when selecting parameters
4. Implement multiple edge detection methods for comprehensive analysis
