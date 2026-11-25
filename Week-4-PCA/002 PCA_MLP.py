import time
import numpy as np

class Dataset:
	def __init__(self):
		self.path = "/storage/emulated/0/DATASETS/handwritten_digit.csv" # change this to your csv path
		self.X = []
		self.Y = []
		self.mean = 0
		self.pc = []
		self.train_X = []
		self.vald_X = []
		self.test_X = []
	
	def load_data(self):
		data = np.genfromtxt(self.path, delimiter = ",", skip_header=0)
		self.X = data[:, 1:] / 255.0
		self.Y = data[:, 0].astype(int)
		self.Y = np.eye(10)[self.Y]
	
	def split_data(self):
		indices = np.random.permutation(len(self.X)) #10000
		limit_1 = int(len(self.X) * 0.7) #7000
		limit_2 = int(len(self.X) * 0.85) #8500
		self.X = self.X[indices]
		self.Y = self.Y[indices]
		self.train_X = self.X[:limit_1]
		self.train_Y = self.Y[:limit_1]
		self.vald_X = self.X[limit_1:limit_2]
		self.vald_Y = self.Y[limit_1:limit_2]
		self.test_X = self.X[limit_2:]
		self.test_Y = self.Y[limit_2:]
	
	def PCA(self):
		self.mean = np.mean(self.train_X, axis=0)
		D = (self.train_X - self.mean)
		cov = np.dot(D.T.copy(), D) / (len(self.train_X) - 1)
		e_val, e_vec = np.linalg.eigh(cov)
		indices = np.argsort(e_val)[::-1]
		e_val = e_val[indices]
		trace = np.sum(e_val)
		evr = e_val / trace
		cumulative_evr = np.cumsum(evr)
		self.n_compo = np.where(cumulative_evr >= 0.95)[0][0] + 1
		self.pc = e_vec[:, indices[:self.n_compo]]
	
	def apply_PCA(self):
		self.train_X = (self.train_X - self.mean) @ self.pc 
		self.vald_X = (self.vald_X - self.mean) @ self.pc
		self.test_X = (self.test_X - self.mean) @ self.pc
	
	def add_bias(self):
		self.train_X = np.hstack((self.train_X, np.ones((len(self.train_X), 1))))
		self.vald_X = np.hstack((self.vald_X, np.ones((len(self.vald_X), 1))))
		self.test_X = np.hstack((self.test_X, np.ones((len(self.test_X), 1))))

class Model:
	def __init__(self, dataset):
		self.train_X = dataset.train_X
		self.train_Y = dataset.train_Y
		self.vald_X = dataset.vald_X
		self.vald_Y = dataset.vald_Y
		self.test_X = dataset.test_X
		self.test_Y = dataset.test_Y
		self.theta_1 = np.random.randn(self.train_X.shape[1], 128)
		self.theta_1 = self.theta_1 * (np.sqrt(2 / self.train_X.shape[1]))
		self.theta_2 = np.random.randn((self.theta_1.shape[1] + 1), self.train_Y.shape[1])
		self.theta_2 = self.theta_2 * 0.01
		self.shuffled_X = []
		self.shuffled_Y = []
		self.batch_size = 33
		self.n = 0
		self.lamb = 0.0001
		self.lr = 0.5
		self.es = 1e-8
	
	def relu(self, z):
		return np.maximum(0, z)
	
	def softmax(self, z):
		exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
		return exp_z / (np.sum(exp_z, axis=1, keepdims=True))
	
	def train_model(self, epochs):
		for epoch in range(epochs):
			indices = np.random.permutation(len(self.train_X)) # 7000
			self.shuffled_X = self.train_X[indices]
			self.shuffled_Y = self.train_Y[indices]
			for i in range(0, len(self.shuffled_X), self.batch_size):
				x = self.shuffled_X[i:i + self.batch_size]
				self.n = len(x)
				y = self.shuffled_Y[i:i + self.batch_size]
				z_1 = x @ self.theta_1
				h_1 = self.relu(z_1)
				h_1 = np.hstack((h_1, np.ones((len(h_1), 1))))
				z_2 = h_1 @ self.theta_2
				h_2 = self.softmax(z_2)
				grad_one = (1 / self.n) * (x.T @ (((h_2 - y) @ self.theta_2.T[:, :-1]) * (z_1 > 0)))
				grad_two = (1 / self.n) * (h_1.T @(h_2 - y))
				grad_one[:-1, :] += self.theta_1[:-1, :] * self.lamb
				grad_two[:-1, :] += self.theta_2[:-1, :] * self.lamb
				self.theta_1 -= self.lr * grad_one
				self.theta_2 -= self.lr * grad_two
			z_1 = self.train_X @ self.theta_1
			h_1 = self.relu(z_1)
			h_1 = np.hstack((h_1, np.ones((len(h_1), 1))))
			z_2 = h_1 @ self.theta_2
			h_2 = self.softmax(z_2)
			penalty = (self.lamb / 2) * (np.linalg.norm(self.theta_1[:-1, :], ord="fro") ** 2 + np.linalg.norm(self.theta_2[:-1, :], ord="fro") ** 2)
			loss = np.mean(-np.sum(self.train_Y * np.log(h_2 + self.es), axis=1)) + penalty
			if epoch % 5 == 0:
				print(f"Epoch: {epoch} Loss: {loss}")
				self.validation()
		pred = np.argmax(h_2, axis=1)
		true = np.argmax(self.train_Y, axis=1)
		accuracy = np.mean(pred == true) * 100
		print(f"\nTraining Accuracy: {accuracy:.2f}")
	
	def validation(self):
		z_1 = self.vald_X @ self.theta_1
		h_1 = self.relu(z_1)
		h_1 = np.hstack((h_1, np.ones((len(h_1), 1))))
		z_2 = h_1 @ self.theta_2
		h_2 = self.softmax(z_2)
		pred = np.argmax(h_2, axis=1)
		true = np.argmax(self.vald_Y, axis=1)
		accuracy = np.mean(pred == true) * 100
		print(f"Validation Accuracy: {accuracy:.2f}")
	
	def evaluate(self):
		z_1 = self.test_X @ self.theta_1
		h_1 = self.relu(z_1)
		h_1 = np.hstack((h_1, np.ones((len(h_1), 1))))
		z_2 = h_1 @ self.theta_2
		h_2 = self.softmax(z_2)
		pred = np.argmax(h_2, axis=1)
		true = np.argmax(self.test_Y, axis=1)
		accuracy = np.mean(pred == true) * 100
		print(f"Test Accuracy: {accuracy:.2f}")

np.random.seed(42)
dataset = Dataset()
dataset.load_data()
dataset.split_data()
ask = input("wants PCA? (y or n): ")
if ask == "y":
	dataset.PCA()
	dataset.apply_PCA()
dataset.add_bias()
print(f"Shape of train X: {dataset.train_X.shape}")
model = Model(dataset)
epochs = 31 #
start = time.time()
model.train_model(epochs)
print("Time per epoch:", (time.time() - start) / epochs)
model.evaluate()