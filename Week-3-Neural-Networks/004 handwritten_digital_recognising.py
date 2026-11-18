import numpy as np

class Dataset:
	def __init__(self):
		self.path = "handwritten_digit.csv"
		self.X = []
		self.Y = []
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []
	
	def load_data(self):
		data = np.genfromtxt(self.path, delimiter = ",", skip_header=0)
		self.X = data[:, 1:] / 255.0
		self.Y = data[:, 0].astype(int)
		self.Y = np.eye(10)[self.Y]
	
	def split_data(self):
		indices = np.random.permutation(len(self.X))
		limit = int(len(self.X) * 0.7)
		v_lim = int(len(self.X) * 0.85)
		self.X = self.X[indices]
		self.Y = self.Y[indices]
		self.train_X = self.X[:limit]
		self.train_X = np.hstack((self.train_X, np.ones((len(self.train_X), 1))))
		self.train_Y = self.Y[:limit]
		self.test_X = self.X[limit:v_lim]
		self.test_X = np.hstack((self.test_X, np.ones((len(self.test_X), 1))))
		self.test_Y = self.Y[limit:v_lim]
		self.validate_X = self.X[v_lim:]
		self.validate_X = np.hstack((self.validate_X, np.ones((len(self.validate_X), 1))))
		self.validate_Y = self.Y[v_lim:]

class Model:
	def __init__(self, dset):
		self.train_X = dset.train_X
		self.train_Y = dset.train_Y
		self.test_X = dset.test_X
		self.test_Y = dset.test_Y
		self.es = 1e-8
		self.train_n = len(self.train_Y)
		self.test_n = len(self.test_Y)
		self.theta_1 = np.random.randn(785, 128) * np.sqrt(2 / 785)
		self.theta_2 = np.random.randn(129, 10) * 0.01
		self.lr = 0.5
		self.batch_size = 33
		self.train_X_shuffled = []
		self.train_Y_shuffled = []
		self.validate_X = dset.validate_X
		self.validate_Y = dset.validate_Y
		self.lam = 0.0001
	
	def relu(self, x):
		return np.maximum(0, x)
	
	def softmax(self, x):
		exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
		return exp_x / np.sum(exp_x, axis=1, keepdims=True)
	
	def train(self, eps):
		for i in range(eps):
			indices = np.random.permutation(len(self.train_X))
			self.train_X_shuffled = self.train_X[indices]
			self.train_Y_shuffled = self.train_Y[indices]
			for j in range(0, len(self.train_X_shuffled), self.batch_size):
				x = self.train_X_shuffled[j:j + self.batch_size]
				y = self.train_Y_shuffled[j:j + self.batch_size]
				m = len(x)
				Z_1 = x @ self.theta_1
				H_1 = self.relu(Z_1)
				H_1 = np.hstack((H_1, np.ones((len(H_1), 1))))
				Z_2 = H_1 @ self.theta_2
				H_2 = self.softmax(Z_2)
				grad_1 = (1 / m) * (x.T @ (((H_2 - y) @ self.theta_2[:-1, :].T) * (Z_1 > 0)))
				grad_2 = (1 / m) * (H_1.T @ (H_2 - y))
				grad_1[:-1, :] += self.lam * self.theta_1[:-1, :]
				grad_2[:-1, :] += self.lam * self.theta_2[:-1, :]
				self.theta_1 -= self.lr * grad_1
				self.theta_2 -= self.lr * grad_2
			Z_1 = self.train_X @ self.theta_1
			H_1 = self.relu(Z_1)
			H_1 = np.hstack((H_1, np.ones((len(H_1), 1))))
			Z_2 = H_1 @ self.theta_2
			H_2 = self.softmax(Z_2)
			penalty = (self.lam / 2) * (np.linalg.norm(self.theta_1[:-1, :], ord="fro") ** 2 + np.linalg.norm(self.theta_2[:-1, :], ord="fro") ** 2)
			loss = np.mean(-np.sum(self.train_Y * np.log(H_2 + self.es), axis=1)) + penalty
			if i % 5 == 0:
				print(f"Epoch: {i} Loss: {loss}")
				self.validation()
		pred = np.argmax(H_2, axis=1)
		true = np.argmax(self.train_Y, axis=1)
		accu = np.mean(pred == true)
		print(f"\nTraining Accuracy: {accu * 100:.2f}%")
	
	def evaluate(self):
		Z_1 = self.test_X @ self.theta_1
		H_1 = self.relu(Z_1)
		H_1 = np.hstack((H_1, np.ones((len(H_1), 1))))
		Z_2 = H_1 @ self.theta_2
		H_2 = self.softmax(Z_2)
		pred = np.argmax(H_2, axis=1)
		true = np.argmax(self.test_Y, axis=1)
		accu = np.mean(pred == true)
		print(f"Test Accuracy: {accu * 100:.2f}")
	
	def validation(self):
		Z_1 = self.validate_X @ self.theta_1
		H_1 = self.relu(Z_1)
		H_1 = np.hstack((H_1, np.ones((len(H_1), 1))))
		Z_2 = H_1 @ self.theta_2
		H_2 = self.softmax(Z_2)
		pred = np.argmax(H_2, axis=1)
		true = np.argmax(self.validate_Y, axis=1)
		accu = np.mean(pred == true)
		print(f"Validation Accuracy: {accu * 100:.2f}")

np.random.seed(0)
dset = Dataset()
dset.load_data()
dset.split_data()
model = Model(dset)
model.train(31)
model.evaluate()