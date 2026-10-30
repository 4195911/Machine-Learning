import numpy as np

np.random.seed(0)
x = np.random.randn(100, 1)
y = 3 * x + 4 + np.random.randn(100, 1)
w = 0.0
b = 0
x = x * 100
x = (x - x.mean()) / x.std()

def forward():
	z = x * w + b
	return z

def train(lr):
	global w, b
	for epoch in range(101):
		z = forward()
		e = z - y
		n = len(y)
		MSE = (1 / n) * np.sum((e)**2)
		gradW = (2 / n) * np.sum(x * (e))
		gradB = (2 / n) * np.sum(e)
		w -= lr * gradW
		b -= lr * gradB
		if epoch % 10 == 0:
			print(f"Epoch: {epoch} Loss: {MSE}")
train(0.1)