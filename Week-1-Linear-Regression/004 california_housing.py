import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

# getting and reshaping data
X = housing.data
y = housing.target
y = y.reshape(-1, 1)

# splitting data
np.random.seed(0)
n_len = len(X)
indices = np.random.permutation(n_len)
perc = int(0.8 * n_len)
X = X[indices]
y = y[indices]
X_train = X[:perc]
y_train = y[:perc]
X_test = X[perc:]
y_test = y[perc:]

# normalizing data
eps = 1e-8
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / (std + eps)
X_test = (X_test - mean) / (std + eps)

# adding bias columns
X_train = np.hstack([X_train, np.ones((len(X_train), 1))])
X_test = np.hstack([X_test, np.ones((len(X_test), 1))])

# initialinzing parameters
theta = np.zeros((9, 1))
lr = 0.1

# training
def training(epochs):
	global theta
	for epoch in range(epochs):
		grad = (2 / len(y_train)) * X_train.T @ (X_train @ theta - y_train)
		theta -= lr * grad
		if epoch % 100 == 0:
			mse = (1 / len(y_train) * np.sum((X_train @ theta - y_train) ** 2))
			print(f"Epoch: {epoch} Loss (MSE): {mse}")
training(501)

# Evaluation
train_pred = X_train @ theta
test_pred= X_test @ theta
train = np.sqrt((1 / len(y_train) * np.sum((train_pred - y_train) ** 2)))
test = np.sqrt((1 / len(y_test) * np.sum((test_pred - y_test) ** 2)))

print("\nTraining (RMSE): ", train)
print("Testing (RMSE): ", test)