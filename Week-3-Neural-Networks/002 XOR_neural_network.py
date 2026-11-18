import numpy as np

np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = np.hstack((X, np.ones((4, 1))))
Y = np.array([[0], [1], [1], [0]])
theta_1 = np.random.randn(3, 2)
theta_2 = np.random.randn(3, 1)
lr = 0.1

def relu(x):
	return np.maximum(0, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train(ep):
	global theta_1, theta_2
	for i in range(ep):
		n = len(Y)
		es = 1e-8
		Z_1 = X @ theta_1
		H_1 = relu(Z_1)
		H_1 = np.hstack((H_1, np.ones((4, 1)))) # (4, 3)
		Z_2 = H_1 @ theta_2
		H_2 = sigmoid(Z_2) # (4, 1)
		loss = np.mean(-((Y * np.log(H_2 + es) + (1 - Y) * np.log(1 - H_2 + es))))
		grad_1 = (1 / n) * X.T @ (((H_2 - Y) @ theta_2.T[:, :2]) * (Z_1 > 0))
		grad_2 = (1 / n) * (H_1.T @ (H_2 - Y))
		theta_1 -= lr * grad_1
		theta_2 -= lr * grad_2
		if i % 100 == 0:
			print(f"Epoch: {i} Loss: {loss}")
	pred = (H_2 > 0.5).astype(float)
	accu = np.mean(pred == Y)
	print(f"\nH_2: {H_2}")
	print(f"\nPrediction: {pred}")
	print(f"\nAccuracy: {accu * 100:.2f}")
	print(f"\nTrained Theta 1: {theta_1}")
	print(f"\nTrained Theta 2: {theta_2}")
train(1001)