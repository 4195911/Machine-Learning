import numpy as np

np.seterr(divide='raise')
class DataSet:
	def __init__(self):
		self.path = "handwritten_digit.csv"
		self.y = [] # (10000, 1)
		self.X = [] # (10000, 784)
		self.X_train = [] # (10000, 784)
		self.y_train = [] # (10000, 1)
		self.X_test = [] # (2000, 784)
		self.y_test = [] # (2000, 1)
	
	def load_data(self):
		data = np.genfromtxt(
			self.path,
			delimiter=',',
			skip_header=0
		)
		samples = data[:, 0]
		self.y = (samples == 0).astype(float).reshape(-1, 1)
		self.X = data[:, 1:]
		return self.y.shape, self.X.shape
	
	def split_data(self):
		indices = np.random.permutation(len(self.X)) # random 10000 indices
		perc = int(len(self.X) * 0.8)
		self.X = self.X[indices]
		self.y = self.y[indices]
		self.X_train = self.X[:perc]
		self.y_train = self.y[:perc]
		self.X_test = self.X[perc:]
		self.y_test = self.y[perc:]
		return self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape
	
	def normalization(self):
		self.X_train = self.X_train / 255.0
		self.X_test = self.X_test / 255.0

class Model:
	def __init__(self, dataset):
		self.dataset = dataset
		self.X_train = self.dataset.X_train
		self.y_train = self.dataset.y_train
		self.X_test = self.dataset.X_test
		self.y_test = self.dataset.y_test
		self.w = np.random.randn(784, 1) * 0.01
		self.b = 0.0
		self.epochs = 1001
		self.eps = 1e-8
		self.n_len = 1 / len(self.y_train)
		self.lr = 0.1
	
	def sigmoid(self, Z):
		return 1 / (1 + np.exp(-Z))
	
	def training(self):
		for epoch in range(self.epochs):
			Z = self.X_train @ self.w + self.b
			p = self.sigmoid(Z)
			Loss = np.mean(-((self.y_train * np.log(p + self.eps)) + ((1 - self.y_train) * np.log(1 - p + self.eps))))
			dL_dz = p - self.y_train # (8000, 1)
			dL_dw = self.n_len * (self.X_train.T @ dL_dz) # (784, 1)
			dL_db = self.n_len * np.sum(p - self.y_train) # (1,)
			self.w = self.w - self.lr * dL_dw
			self.b = self.b - self.lr * dL_db
			if epoch % 100 == 0:
				print(f"Epoch: {epoch} Loss: {Loss}")
	
	def evaluate(self):
		Z = self.X_test @ self.w + self.b
		p = self.sigmoid(Z) # (2000, 1)
		pred = (p >= 0.5).astype(int)
		accuracy = (np.sum(pred == self.y_test)) / len(self.y_test)
		print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

np.random.seed(2)
dataset = DataSet()
dataset.load_data()
dataset.split_data()
dataset.normalization()
model = Model(dataset)
model.training()
model.evaluate()