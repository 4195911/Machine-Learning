import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.randn(10, 1)
y = 3 * x + 4 + np.random.randn(10, 1)
data = np.hstack((x, y))

#feature_A = np.mean(data[:, 0])
#feature_B = np.mean(data[:, 1])

#deviation_A = data[:, 0] - feature_A # 10,
#deviation_B = data[:, 1] - feature_B # 10,
#n = len(data)

#var_A = (np.sum(deviation_A ** 2)) / (n - 1)
#var_B = (np.sum(deviation_B ** 2)) / (n - 1)
#cov_AB = (np.sum(deviation_A * deviation_B)) / (n - 1)

#cov_matrix = np.array([[var_A, cov_AB], [cov_AB, var_B]])
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

mean = np.mean(data, axis=0)
D = data - mean # 10, 2
idx = np.argmax(eigenvalues)
pc1 = eigenvectors[:, idx]
Z = D @ pc1 # 10,

projected = np.outer(Z, pc1) + mean
t = np.linspace(Z.min(), Z.max(), 100)
pc1_line = mean + np.outer(t, pc1)

plt.figure(figsize=(7, 5))
plt.scatter(data[:, 0], data[:, 1], label="Original Points")
plt.scatter(projected[:, 0], projected[:, 1], color="red", label="Projected Points")
plt.plot(pc1_line[:, 0], pc1_line[:, 1], color="blue", label="PC1 direction")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()