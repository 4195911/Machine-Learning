import numpy as np

x = np.random.randn(100, 1)
X = np.hstack([x, np.ones((100, 1))])

y = 3 * x + 4 + np.random.randn(100, 1)

theta = np.zeros((2, 1))
lr = 0.1

for epoch in range(100001):
	grad = (2 / len(y)) * X.T @ (X @ theta - y)
	theta -= lr * grad
	print(theta)