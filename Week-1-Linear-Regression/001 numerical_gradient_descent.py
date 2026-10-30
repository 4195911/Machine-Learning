import numpy as np

np.random.seed(0)
x = np.random.randn(100, 1)
y = 3 * x + 4 + np.random.randn(100, 1)

def mse(w, b):
	z = x * w + b
	e = z - y
	l2 = np.linalg.norm(e)
	loss = (l2 ** 2) / len(e)
	return loss

lr = 0.1
w = 0.0
b = 0
h = 0.01

for epoch in range(300):
	d_w = (mse(w + h, b) - mse(w, b)) / h
	d_b = (mse(w, b + h) - mse(w, b)) / h
	w -= lr * d_w
	b -= lr * d_b
	if epoch % 3 == 0:
		print(f"Epoch: {epoch} | weights: {w} | Bias: {b} | Loss: {mse(w, b)}")