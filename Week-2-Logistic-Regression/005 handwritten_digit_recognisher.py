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
		data = np.genfromtxt(
			self.path,
			delimiter = ",",
			skip_header=0
		)
		
		self.X = data[:, 1:] / 255.0
		self.Y = data[:, 0].astype(int)
		self.Y = np.eye(10)[self.Y]
	
	def split_data(self):
		indices = np.random.permutation(len(self.X))
		perc = int(len(self.X) * 0.8)
		
		self.X = self.X[indices]
		self.Y = self.Y[indices]
		
		self.train_X = self.X[:perc]
		self.train_X = np.hstack((self.train_X, np.ones((8000, 1))))
		self.train_Y = self.Y[:perc]
		
		self.test_X = self.X[perc:]
		self.test_X = np.hstack((self.test_X, np.ones((2000, 1))))
		self.test_Y = self.Y[perc:]

class Model:
	def __init__(self, dataset):
		self.train_X = dataset.train_X
		self.train_Y = dataset.train_Y
		self.test_X = dataset.test_X
		self.test_Y = dataset.test_Y
		self.theta = np.zeros((785, 10))
		self.lr = 1.5
		
	def softmax(self, Z):
		exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
		return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
	
	def train_model(self):
		for i in range(101):
			Z = self.train_X @ self.theta
			p = self.softmax(Z)
			loss = -((1 / len(self.train_Y)) * np.sum(self.train_Y * np.log(p + 1e-8)))
			grad_theta = (1 / len(self.train_Y) * self.train_X.T @ (p - self.train_Y))
			self.theta = self.theta - self.lr * grad_theta
			if i % 10 == 0:
				print(f"Epoch: {i} Loss: {loss}")
		pred = np.argmax(p, axis=1)
		acc = np.mean(pred == np.argmax(self.train_Y, axis=1))
		print(f"Training Accuracy: {acc * 100:.2f}%")
	
	def evaluate(self):
		Z = self.test_X @ self.theta
		p = self.softmax(Z)
		pred = np.argmax(p, axis=1)
		true = np.argmax(self.test_Y, axis=1)
		acc = np.mean(pred == true)
		print(f"Test Accuracy: {acc * 100:.2f}%")

np.random.seed(0)
dset = Dataset()
dset.load_data()
dset.split_data()
model = Model(dset)
model.train_model()
model.evaluate()