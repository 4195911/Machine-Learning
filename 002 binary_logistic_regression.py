import numpy as np

np.random.seed(0)
x = np.random.randn(100, 1)
y = 3 * x + 4 + np.random.randn(100, 1)
lables = (y > 4).astype(float) # 0 or 1

w = np.zeros((1, 1))
b = 0
lr = 0.1

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def forward(w, b):
	Z = x @ w + b
	p = sigmoid(Z)
	return p # (100, 1)

def training(epochs):
	global w, b
	for epoch in range(epochs):
		p = forward(w, b)
		Loss = np.mean(-(((lables) * np.log(p + 1e-8)) + ((1 - lables) * np.log(1 - p + 1e-8))))
		dL_dz = p - lables # (100, 1)
		dL_dw = (1 / len(lables)) * (x.T @ (dL_dz)) # (1, 1)
		dL_db = (1 / len(lables)) * np.sum(p - lables) # (1,)
		w -= lr * dL_dw
		b -= lr * dL_db
		if epoch % 10 == 0:
			print(
			    f"Epoch: {epoch:4d} | "
			    f"Loss: {Loss:.4f} | "
			    f"Weight: {w.item():.4f} | "
			    f"Bias: {b:.4f} | "
			    f"dL/dw: {dL_dw.item():.4f} | "
			    f"dL/db: {dL_db:.4f}"
			)
training(1001)