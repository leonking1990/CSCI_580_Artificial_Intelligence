import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('linear_regression_data.csv', header=None, names=['x', 'y'])

# Extract x and y values
x = data['x'].values
y = data['y'].values

# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate covariance and variance
cov_xy = np.sum((x - mean_x) * (y - mean_y)) / len(x)
var_x = np.sum((x - mean_x) ** 2) / len(x)

# Calculate slope (m) and intercept (b)
m = cov_xy / var_x
b = mean_y - m * mean_x

# Predict y values using the linear model
y_pred = m * x + b

# Plot the data points and the linear regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label=f'Linear Model: y = {m:.2f}x + {b:.2f}')
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (y)')
plt.title('Linear Regression using Covariance Approach')
plt.legend()
plt.grid(True)
plt.show()
