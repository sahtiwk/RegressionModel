# Linear Regression Models from Scratch

![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)

## 📖 Overview

This repository contains implementations of **Simple Linear Regression** and **Multiple Linear Regression** models built from the ground up using Python and NumPy. The primary goal of this project is to demonstrate a deep understanding of the mathematical principles behind these fundamental machine learning algorithms, moving beyond just using pre-built libraries like Scikit-learn.

---

## ✨ Key Features

* **Simple Linear Regression:** A model to find the relationship between a single feature and a target variable.
* **Multiple Linear Regression:** An advanced model that uses the Normal Equation to handle multiple input features.
* **Built from Scratch:** Core logic is implemented using only NumPy for numerical operations.
* **Standard ML Interface:** Classes include the standard `.fit()`, `.predict()`, and `.score()` methods.
* **Demonstrated on Real Datasets:** The models are tested and validated on standard datasets.

---

## 🚀 Models Implemented

### 1. Simple Linear Regression

This model calculates the optimal slope and intercept to fit a straight line through data points using the method of least squares. It's ideal for understanding the relationship between two continuous variables.

#### Example Usage: Predicting Salary from Experience

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming the LinearRegression class is defined in your notebook/script
# from linear_regression import LinearRegression

# Load the dataset
salary_df = pd.read_csv('Salary_dataset.csv')
X = np.array(salary_df['YearsExperience'])
y = np.array(salary_df['Salary'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model R-squared Score: {score:.4f} ({score:.2%})")

# --- Output ---
# Model Trained
# Slope and Intercept:  9423.815323030978 24380.201479473697
# Model R-squared Score: 0.9024 (90.24%)
```

### 2. Multiple Linear Regression

This model uses the **Normal Equation**—a linear algebra technique—to solve for the optimal coefficients for multiple input features at once. It can capture more complex relationships by considering how multiple variables collectively influence the target.

#### Example Usage: Predicting California Housing Prices

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming the MultipleLinearRegression class is defined
# from multiple_linear_regression import MultipleLinearRegression

# Load and prepare data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
multi_model = MultipleLinearRegression()
multi_model.fit(X_train_scaled, y_train)

# Evaluate the model
score = multi_model.score(X_test_scaled, y_test)
print(f"Model R-squared score: {score:.4f}")

# --- Output ---
# Model Trained.
# Model R-squared score: 0.5888
```

---

## 🛠️ Getting Started

### Prerequisites

Make sure you have Python 3 installed. You will need the following libraries:
* NumPy
* Pandas
* Scikit-learn
* Seaborn

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install dependencies:**
    It's recommended to create a `requirements.txt` file with the following content:
    ```txt
    numpy
    pandas
    scikit-learn
    seaborn
    ```
    Then, install them using pip:
    ```sh
    pip install -r requirements.txt
    ```

---

## 💡 Future Improvements

This project serves as a strong foundation. Future enhancements could include:

* **Gradient Descent:** Implementing an iterative optimization algorithm to train the models, which is more scalable for very large datasets.
* **Regularization:** Adding L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
* **Additional Metrics:** Including other evaluation metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
* **Unified Class:** Combining both models into a single, robust `LinearRegression` class that can handle both simple and multiple regression cases.

---
