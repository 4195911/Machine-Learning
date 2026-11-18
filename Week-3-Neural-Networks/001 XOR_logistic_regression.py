import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4×2
y = np.array([[0], [1], [1], [0]]) # 4×1

X= np.hstack((x, np.ones((4, 1)))) # 4×3

theta = np.zeros((3, 1))
lr = 0.1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train(epochs):
	global theta
	for epoch in range(epochs):
		Z = X @ theta
		p = sigmoid(Z)
		loss = np.mean(-((y * np.log(p + 1e-8)) + ((1 - y) * np.log(1 - p + 1e-8))))
		grad = (1 / len(y)) * X.T @ (p - y)
		theta = theta - lr * grad
		if epoch % 10 == 0:
			print(f"Epoch: {epoch} Loss: {loss}")
	pred = p > 0.5
	accu = np.mean(pred == y) * 100
	print(f"\nTraining Accuracy: {accu:.2f}%")

train(101)