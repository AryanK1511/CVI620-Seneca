# CVI Midterm

### **Part A: Image Acquisition and Preprocessing**

#### **Task 1: Acquiring Sample Images**

**Code:**

```python
import cv2
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('sample_image.jpg')

# Display using OpenCV
cv2.imshow('Original Image - OpenCV', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image - Matplotlib')
plt.show()
```

**Deliverable:** You will need to submit original images and highlight different types of edges in those images, such as sharp, soft, and complex.

#### **Task 2: Grayscale Conversion**

**Code:**

```python
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.show()
```

**Why Grayscale?** Grayscale simplifies edge detection by reducing the computational complexity and focusing on intensity changes instead of color.

#### **Task 3: Noise Reduction Using Smoothing**

**Code:**

```python
# Apply Gaussian Blur for noise reduction
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Display the original and blurred image for comparison
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Blurred')
plt.show()
```

**Deliverable:** Submit the code, and display images before and after applying smoothing filters like Gaussian Blur or Median Filter.

---

### **Part B: Edge Detection Techniques**

#### **Task 4: Sobel Edge Detection**

**Code:**

```python
# Sobel Edge Detection in X and Y direction
sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=5)

# Combine Sobel X and Y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Display Sobel result
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.show()
```

**Deliverable:** Provide the code and display the edge-detected image using Sobel, explaining how it calculates gradients in both X and Y directions.

#### **Task 5: Laplacian Edge Detection**

**Code:**

```python
# Laplacian Edge Detection
laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_64F)

# Display Laplacian result
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.show()
```

**Deliverable:** Provide the code and display the Laplacian-detected edges.

#### **Task 7: Canny Edge Detection**

**Code:**

```python
# Canny Edge Detection
canny_edges = cv2.Canny(gaussian_blur, 100, 200)

# Display Canny result
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
```

**Deliverable:** Submit the code and display the Canny edge-detected image. Experiment with different thresholds and explain why Canny is robust compared to Sobel and Laplacian.

---

### **Part C: Performance Analysis and Optimization**

#### **Task 8: Comparing Edge Detection Techniques**

**Code for Comparison:**

```python
# Display comparison of Sobel, Laplacian, and Canny edge detection
plt.figure(figsize=(10, 7))

# Sobel
plt.subplot(1, 3, 1)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel')

# Laplacian
plt.subplot(1, 3, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')

# Canny
plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny')

plt.show()
```

**Deliverable:**

- Create a comparison report, including edge sharpness, accuracy, noise sensitivity, and computational efficiency.
- Provide visual examples and summarize the pros and cons of each method in a table.

---

### **Final Report Outline**

1. **Introduction**  
   An overview of edge detection and its applications in image processing and computer vision.

2. **Methods**  
   Detailed explanation of Sobel, Laplacian, and Canny edge detection with code snippets.

3. **Results**  
   Present edge-detected images using each method, along with an analysis of the performance.

4. **Challenges**  
   Discuss any challenges faced during implementation (e.g., handling noise, tuning thresholds) and how they were addressed.

5. **Conclusion**  
   Summary of findings, comparison between methods, and suggestions for future improvements in edge detection.

---

### **Grading Criteria Breakdown**

- **Code Functionality:** Make sure the edge detection algorithms perform correctly and efficiently.
- **Report Quality:** Clear explanations of each method and result with well-documented code.
- **Creativity and Problem-Solving:** Explore different parameters and optimizations in edge detection techniques.
- **Presentation:** Ensure high-quality visuals and structured content in the final report.
