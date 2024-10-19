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

![img](https://drive.google.com/thumbnail?id=1S0RWehydZDr2RAtXP7ZcsgJt4a8KyJeg&sz=w1000)

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

![img](https://drive.google.com/thumbnail?id=1c1TtuU9LV44avLJq_l0Gy6LRSTKkAi8h&sz=w1000)

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

![img](https://drive.google.com/thumbnail?id=1fIlAmoAGJetEdsBXce-T7_OF69N1hWyr&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1qk6n-PYfDQErC5GlkFxseM67KGHQm8tT&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1r6jMfItyQy4xnNdR6fGyhvVILUYA3OG7&sz=w1000)

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

![img](https://drive.google.com/thumbnail?id=1jsXluBdSjGeYjxhCw5HfvwjJ4WB9ciEe&sz=w1000)

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

![img](https://drive.google.com/thumbnail?id=124D2vKvUTKcn-aMYHNtm7n-sRR4Vm7hE&sz=w1000)

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

![img](https://drive.google.com/thumbnail?id=1EjvDK4Dn97Mvg31snaCdiignmFIpYxe4&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1N0woWi_Kq1LgDsfv4nDDkEcjCwjc-ULZ&sz=w1000)
![img](https://drive.google.com/thumbnail?id=1t0jmK9JKGEaZ8jjiUZ06I_3fN3fH_g3g&sz=w1000)

### Task 7: Canny Edge Detection

- Canny edge detection algorithm begins by applying a Gaussian filter to smooth the image, which helps reduce noise that can interfere with edge detection.
- Following this, it calculates the intensity gradient using methods like the Sobel operator, highlighting areas where the intensity changes rapidly, which indicates the presence of edges.
- The next step is non-maximum suppression, where the algorithm thins out detected edges to a single pixel width, enhancing the clarity of the edges.
- This combination of techniques allows Canny edge detection to excel in identifying complex and curved edges while maintaining sensitivity to significant features.

```python
# Initialize a list to store images with Canny edge detection results.
canny_images = []

# Define a list of threshold pairs for Canny edge detection.
thresholds = [(50, 150), (100, 200), (150, 250)]

# Loop through each blurred image and its index.
for index, blurred_image in enumerate(blurred_images):
    # Loop through each threshold pair.
    for low_thresh, high_thresh in thresholds:
        # Apply Canny edge detection using the current threshold pair.
        canny_edges = cv.Canny(blurred_image, low_thresh, high_thresh)

        # Append the Canny edge-detected image to the list if the low threshold is 50.
        if low_thresh == 50:
            canny_images.append(canny_edges)

        # Display the Canny edge-detected image with a window name indicating the image and thresholds used.
        cv.imshow(f"Canny Edge Detection - Image {index + 1} (Thresh: {low_thresh}, {high_thresh})", canny_edges)

# Wait for a key press to close the displayed windows.
cv.waitKey(0)
cv.destroyAllWindows()

```

#### Different Thresholds

- The **low threshold** is used to identify weak edges. Any pixel with a gradient intensity below this threshold is discarded and not considered part of an edge.
- The **high threshold** is set to identify strong edges. Pixels with gradient values above this threshold are classified as strong edges, indicating clear and significant transitions in intensity. These strong edges are guaranteed to be retained in the final edge-detected image.
- Pixels that fall between the low and high thresholds are classified as weak edges. Their status is conditional; they are retained only if they are connected to strong edges. This means that weak edges can be influential in the final output if they help form a continuous edge along with strong edges.

#### Results of Experimenting with Different Thresholds

- **Lower Thresholds (like 50)**: More inclusive, capturing more detail but potentially including noise.
- **Medium Thresholds (like 100)**: Strikes a balance, filtering some noise while still retaining a reasonable amount of detail.
- **Higher Thresholds (like 150)**: Focuses on strong, significant edges, potentially losing finer details but producing cleaner results.

Here is the output showing an image with different thresholds:

![img](https://drive.google.com/thumbnail?id=19q2KBwuW_ZoiEAJETa_7ZcH8W4WN0Lft&sz=w1000)

#### Comparison of Edge Detection Techniques based on performance and accuracy

1. **Sobel Operator**:

   - **Performance**: Generally faster than more complex techniques, as it involves simple convolution operations.
   - **Accuracy**: Good for detecting edges in images with clear transitions but can be sensitive to noise, which may lead to false edges in cluttered images.

2. **Laplacian Operator**:

   - **Performance**: Slightly slower than Sobel due to second-order derivative calculations. It is still efficient for real-time applications.
   - **Accuracy**: Effective in detecting edges and contours, but can be more prone to noise. It may produce thicker edges compared to Sobel, which can obscure fine details.

3. **Canny Edge Detection**:
   - **Performance**: More computationally intensive due to multiple steps (smoothing, gradient calculation, non-maximum suppression, and hysteresis). However, it is efficient for its accuracy in edge detection.
   - **Accuracy**: Known for high accuracy in detecting edges with better control over noise and false edges. It is less sensitive to noise compared to Sobel and Laplacian, making it ideal for complex images.

While Sobel and Laplacian are quicker and simpler, Canny offers superior accuracy and noise management, making it a preferred choice for more detailed and intricate edge detection tasks. The trade-off often comes down to the specific requirements of the application, such as speed versus the need for precision in edge identification.

| Feature                      | Sobel Operator                                                                        | Laplacian Operator                                                               | Canny Edge Detection                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Edge Sharpness**           | Moderate; detects edges well but can produce thicker edges due to gradient averaging. | Moderate to high; produces sharp edges but may include more noise.               | High; sharp and clean edges due to effective noise reduction and non-maximum suppression.                |
| **Accuracy**                 | Good for clear transitions; can miss fine edges in noisy images.                      | Effective for contour detection; may confuse noise with edges.                   | Very high; excels in detecting actual edges while minimizing false positives.                            |
| **Noise Sensitivity**        | Sensitive to noise; noisy images can lead to false edges.                             | More prone to noise; may detect noise as edges, resulting in less clean outputs. | Less sensitive; effectively reduces noise through smoothing and hysteresis.                              |
| **Computational Efficiency** | Fast; involves simple convolution operations.                                         | Efficient but slightly slower than Sobel due to second-order derivatives.        | More computationally intensive due to multiple processing steps (smoothing, gradient calculation, etc.). |

## Part C: Performance Analysis and Optimization

### Task 8: Comparing Edge Detection Techniques

**Here are some images that compare edge detection results among Sobel, Laplacian, and Canny methods. The images are arranged from left to right, starting with the Sobel method.**

#### IMAGE 01: Geometric Shapes

![img](https://drive.google.com/thumbnail?id=1-uaV6LISiSMrUWrEW4S_h8-P8NAktBC8&sz=w1000)

#### IMAGE 02: Industrial Objects

![img](https://drive.google.com/thumbnail?id=1O1qf-4WLerDW5EiBcs0XPOHsPDs73ZwA&sz=w1000)

#### IMAGE 03: Natural Scenes

![img](https://drive.google.com/thumbnail?id=1D0PfCH-pTLXnsviWabMRVr-RxRffrmtC&sz=w1000)

#### 1. Sobel Operator

- The Sobel operator is effective for detecting edges in images with clear transitions.
- It provides moderate edge sharpness, as it typically produces thicker edges due to the averaging of gradients.
- While it is relatively fast and computationally efficient, the Sobel operator is sensitive to noise, which can lead to false edges in cluttered or noisy images.
- Overall, it strikes a balance between performance and accuracy but may not be ideal for images with a lot of detail or noise.

#### 2. Laplacian Operator

- The Laplacian operator is known for its ability to detect contours effectively, producing sharp edges in many cases.
- However, it can be more prone to noise, often detecting noise as edges, which results in less clean outputs.
- While it operates efficiently, it can be slightly slower than the Sobel operator due to its reliance on second-order derivatives.
- It offers good performance for contour detection but requires careful consideration in noisy environments.

#### 3. Canny Edge Detection

- Canny edge detection is renowned for its high accuracy and ability to produce sharp, clean edges.
- It excels at minimizing false positives and is less sensitive to noise compared to the other two methods, making it highly effective for complex images.
- However, Canny is more computationally intensive because it involves multiple steps.
- Despite the higher computational cost, its superior performance in edge detection often justifies its use in applications where detail and accuracy are critical.

#### Pros and Cons

| Method                   | Pros                                                                    | Cons                                                                       |
| ------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Sobel Operator**       | - Fast and simple and Effective for basic edge detection                | - Sensitive to noise and May produce thicker edges                         |
| **Laplacian Operator**   | - Good for contour detection and Can detect edges in various directions | - More prone to noise and Thicker edges can obscure details                |
| **Canny Edge Detection** | - High accuracy and sharp edges and Robust against noise                | - More computationally intensive and Requires careful tuning of thresholds |

#### A short observation summary

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
4. - Provided the most complete and clean edge maps
   - Different threshold combinations affected results:
     - (50, 150): Detected more edges but included some noise
     - (100, 200): Balanced edge detection
     - (150, 250): Focused on strongest edges only

## Challenges

The challenges in computer vision often involve selecting the perfect kernel size and other parameters for different types of functions.

This is because, in many cases, different parameters work for different images, and there is no "one-size-fits-all" approach when using OpenCV and convolutions. The only method that works for me is trial and error, along with reviewing documentation.

For example, understanding how kernel size works can help you tune your parameters. If I know that a kernel size of 7 would be too large, I might test sizes between 3 and 5.

**My usual workflow for tuning parameters is as follows:**

1. Understand what the parameters do.
2. Determine a reasonable range for the parameters.
3. Test every value within that range.

## Conclusion

### Findings

1. Canny edge detection provided the most reliable results across various image types.
2. Preprocessing steps, such as grayscale conversion and Gaussian blur, were crucial for achieving good results.
3. Parameter selection significantly impacts the quality of edge detection.
4. In computer science, it is always necessary to consider trade-offs; for instance, while Canny may be the best method, we need to evaluate whether it’s worth the resources it requires.

### Future Improvements

1. Implement adaptive parameter selection based on image characteristics. As mentioned earlier, there is no one-size-fits-all solution, so ideally, I should use different parameters for different images. Perhaps the hashmap initialized at the start could also include associated parameters along with the image path.
2. Develop automated evaluation metrics for edge detection quality. Currently, we manually check which result looks best, which may not be the most efficient approach in an industry setting.
3. Although we do not have a proper application right now, considering the specific requirements of an application when selecting parameters could be beneficial if we develop one in the future.
