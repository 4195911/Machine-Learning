import numpy as np

y = np.array([1, 0, 1])
p = np.array([0.99, 0.01, 0.01])

def bce(y, p):
	return np.mean((-((y * np.log(p)) + ((1 - y) * np.log(1 - p)))))
print(bce(y, p))

#[0.01005034, 0.01005034, 4.60517019]

#1.5417569525650314 (mean of loss)