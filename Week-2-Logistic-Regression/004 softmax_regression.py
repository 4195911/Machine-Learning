import numpy as np

np.random.seed(0)
a = np.random.randn(100, 2) + np.array([0, 2])
b = np.random.randn(100, 2) + np.array([-2, -1])
c = np.random.randn(100, 2) + np.array([2, -1])

x = np.vstack((a, b, c))
labels = np.repeat([0, 1, 2], 100).reshape(-1, 1)
Y = np.zeros((300, 3))
Y[np.arange(300), labels.ravel()] = 1
X = np.hstack((x, np.ones((300, 1))))
theta = np.zeros((3, 3))
lr = 0.1

def softmax(Z):
	exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
	return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def train():
	global theta
	for epoch in range(1001):
		Z = X @ theta # 300, 3
		p = softmax(Z) # 300, 3
		loss = -(1 / len(Y)) * (np.sum(Y * np.log(p + 1e-8))) # CCE
		grad_theta = (1 / len(Y)) * X.T @ (p - Y) # (3, 3)
		theta -= lr * grad_theta
		
		if epoch % 100 == 0:
			print(f"Epoch: {epoch} Loss: {loss}")
train()

t_p = softmax(X @ theta)
t_pred = np.argmax(t_p, axis=1)
t_accu = np.mean(t_pred == labels.ravel())
print(f"\nTraining Accuracy: {t_accu * 100:.2f}%")

def test():
	np.random.seed(42)
	a = np.random.randn(100, 2) + np.array([0, 2])
	b = np.random.randn(100, 2) + np.array([-2, -1])
	c = np.random.randn(100, 2) + np.array([2, -1])
	
	x = np.vstack((a, b, c)) # 300, 2
	X = np.hstack((x, np.ones((300, 1))))
	Z = X @ theta
	p = softmax(Z)
	pred = np.argmax(p, axis=1)
	labels = np.repeat([0, 1, 2], 100) # 300,
	accu = np.mean(pred == labels.ravel())
	print(f"Test Accuracy: {accu * 100:.2f}%")
test()