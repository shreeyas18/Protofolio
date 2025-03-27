import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data points resembling the desired distribution
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.zeros(100)
y[X[:, 0] > 0] = 1
X[y == 1] += 2

# Plot the distribution of data points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('Distribution of data points in neural networks activity 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Train the neural network models with specified settings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# No hidden layer
clf_no_hidden = MLPClassifier(hidden_layer_sizes=(), random_state=1, max_iter=1000)
# One hidden layer with two neurons
clf_one_hidden = MLPClassifier(hidden_layer_sizes=(2,), random_state=1, max_iter=1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the models
clf_no_hidden.fit(X_train, y_train)
clf_one_hidden.fit(X_train, y_train)

# Test the models
test_error_no_hidden = 1 - clf_no_hidden.score(X_test, y_test)
test_error_one_hidden = 1 - clf_one_hidden.score(X_test, y_test)

# Report the test error
print("Test Error (No Hidden Layer):", test_error_no_hidden)
print("Test Error (One Hidden Layer with Two Neurons):", test_error_one_hidden)
