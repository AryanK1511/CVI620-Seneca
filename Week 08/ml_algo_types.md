# Different Machine Learning Algorithms

The notes here are based upon [this video](https://www.youtube.com/watch?v=E0Hmnixke2g) that I found on YouTube.

## Broad Machine Learning Categories

1. **Supervised Learning**:

   - **Definition**: Algorithms learn from labeled data (features → target).
   - **Examples**:
     - Predict house prices based on features.
     - Categorize objects (e.g., cat/dog).
   - **Types**:
     - **Regression**: Predict numeric (continuous) outcomes (e.g., house prices).
     - **Classification**: Predict categorical outcomes (e.g., spam/no spam).

2. **Unsupervised Learning**:
   - **Definition**: No labeled data; the algorithm identifies patterns.
   - **Examples**:
     - Group emails into clusters.

## Supervised Learning Algorithms

1. **Linear Regression**:

   - Fits a line to model the relationship between input and output.
   - Goal: Minimize prediction error by reducing the distance to the regression line.

2. **Logistic Regression**:

   - A classification algorithm using a sigmoid function to predict probabilities.
   - Example: Predict gender based on height and weight.

3. **K-Nearest Neighbors (KNN)**:

   - Non-parametric; uses nearby data points for predictions.
   - Works for both regression (average of neighbors) and classification (majority class).
   - **Hyperparameter**: `K` (number of neighbors).

4. **Support Vector Machine (SVM)**:

   - Classifies data by maximizing the margin between classes.
   - Can use kernel functions for complex, non-linear boundaries.
     - Example kernels: Linear, Polynomial, RBF.

5. **Naive Bayes**:

   - Based on Bayes' theorem; assumes feature independence.
   - Common use case: Spam detection.

6. **Decision Trees**:

   - Splits data using yes/no questions to classify or predict.
   - Goal: Maximize "purity" of leaf nodes.

7. **Ensemble Methods**:
   - **Random Forests**: Combines multiple decision trees (bagging).
   - **Boosted Trees**: Sequentially trains models to correct errors (boosting).
     - Examples: AdaBoost, Gradient Boosting, XGBoost.

## Neural Networks (NN)

1. **Overview**:

   - Automatically learns features for classification or regression tasks.
   - Uses layers of nodes:
     - **Input layer**: Raw features.
     - **Hidden layers**: Extract features.
     - **Output layer**: Predictions.
   - Example: Digit classification by recognizing patterns (e.g., vertical lines for "1").

2. **Strengths**:
   - Effective for high-dimensional and complex data.
   - Removes the need for manual feature engineering.

## Intuition & Decision

- **Simple Relationships**: Use Linear/Logistic Regression.
- **Complex Patterns**: Use SVMs, Random Forests, or Boosted Trees.
- **High-Dimensional Data**: Use Neural Networks.
- **Quick & Efficient**: Consider Naive Bayes or KNN for small datasets.
