import numpy as np

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

theta_1 = np.array([
    [ 2.25465261, -2.44832459],
    [-2.25632475,  2.44653015],
    [ 0.00019538, -0.00233675]
])

theta_2 = np.array([
    [ 3.44254991],
    [ 3.18197242],
    [-3.23870705]
])

h = es = 1e-5

lr = 0.1

def relu(x):
	return np.maximum(0, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def loss(theta_1, theta_2):
	Z_1 = X @ theta_1
	H_1 = relu(Z_1)
	H_1 = np.hstack((H_1, np.ones((4, 1))))
	Z_2 = H_1 @ theta_2
	H_2 = sigmoid(Z_2)
	loss = np.mean(-((Y * np.log(H_2 + es) + (1 - Y) * np.log(1 - H_2 + es))))
	return loss

def theta_one(h, i, j):
	theta_1[i][j] += h
	a = loss(theta_1, theta_2)
	theta_1[i][j] -= (2 * h)
	b = loss(theta_1, theta_2)
	theta_1[i][j] += h
	return (a - b)  / (2 * h)

def theta_two(h, i, j):
	theta_2[i][j] += h
	a = loss(theta_1, theta_2)
	theta_2[i][j] -= (2 * h)
	b = loss(theta_1, theta_2)
	theta_2[i][j] += h
	return (a - b)  / (2 * h)

def gradient_1(i, j):
	n = len(Y)
	es = 1e-8
	Z_1 = X @ theta_1
	H_1 = relu(Z_1)
	H_1 = np.hstack((H_1, np.ones((4, 1)))) # (4, 3)
	Z_2 = H_1 @ theta_2
	H_2 = sigmoid(Z_2) # (4, 1)
#	print(X.T.shape, H_2.shape, theta_2.T[:, :2].shape)
	return ((1 / n) * X.T @ (((H_2 - Y) @ theta_2.T[:, :2]) * (Z_1 > 0)))[i][j]

def gradient_2(i, j):
	n = len(Y)
	es = 1e-8
	Z_1 = X @ theta_1
	H_1 = relu(Z_1)
	H_1 = np.hstack((H_1, np.ones((4, 1)))) # (4, 3)
	Z_2 = H_1 @ theta_2
	H_2 = sigmoid(Z_2)
	return ((1 / n) * (H_1.T @ (H_2 - Y)))[i][j]

x, y = theta_1.shape
for i in range(x):
	for j in range(y):
		one = theta_one(h, i, j)
		grad_one = gradient_1(i, j)
		print(
		    f"Weight one: [{i}, {j}] | "
		    f"Numerical: {one:.10f} | "
		    f"Analytical: {grad_one:.10f}"
		)

print("=" * 50)

x, y = theta_2.shape
for i in range(x):
	for j in range(y):
		two = theta_two(h, i, j)
		grad_two = gradient_2(i, j)
		print(
		    f"Weight two: [{i}, {j}] | "
		    f"Numerical: {two:.10f} | "
		    f"Analytical: {grad_two:.10f}"
		)