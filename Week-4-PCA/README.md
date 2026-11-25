Week 4 — Dimensionality Reduction (PCA)

Implemented Principal Component Analysis (PCA) and compared a neural network trained on PCA-transformed features against one trained on raw pixel data.

Projects

- 001 PCA — Implemented PCA using covariance matrices, eigenvalues, and eigenvectors. Projected the original data onto the principal component directions.
- 002 PCA + Neural Network — Combined PCA with a two-layer neural network and compared its performance against the same network trained on raw pixel inputs.

Method

- Dataset: 10,000 handwritten digit images.
- Input features: 784 pixels per image (28 × 28).
- Data split: 70% training, 15% validation, 15% testing.
- PCA: Retained 95% of the variance, reducing the input dimensions from 784 to 148.
- Model: 128 hidden ReLU neurons and 10 softmax output neurons.
- Training: Mini-batch gradient descent, batch size 33, learning rate 0.5, and L2 regularization.
- PCA was fitted using the training set only to prevent data leakage.

Results — PCA vs Raw Pixels

Comparison across three random seeds:

| Metric | PCA (148 features) | Raw Pixels (784 features) |
|---|---:|---:|
| Average validation accuracy | 95.89% | 96.18% |
| Average test accuracy | 95.46% | 95.58% |
| Average seconds per epoch | 19.75 | 65.14 |

- PCA was approximately 3.3× faster during training.
- Test accuracy differed by only 0.12 percentage points on average.
- The accuracy difference changed direction across random seeds, so no meaningful accuracy advantage was established.
- Training accuracy reached 100% in every run, while validation accuracy remained around 96%.

Earlier Experiments

- Learning-rate experiments: 0.5 performed best among the tested values (0.1, 0.5, and 0.03).
- L2 regularization: Lambda values of 0.0001 and 0.001 did not improve validation performance over the baseline.
- Lambda 0.1 was too strong and caused training performance to collapse.

Conclusion

PCA reduced the neural network's input dimensions by approximately 81% and made training about 3.3 times faster, while maintaining comparable test accuracy.

However, PCA did not solve the model's overfitting problem. Further experiments are needed, including early stopping, dropout, data augmentation, and learning curves with different training-set sizes.

Dataset

- MNIST handwritten digits in CSV format.
- Download "train.csv" from the "Kaggle Digit Recognizer dataset" (https://www.kaggle.com/competitions/digit-recognizer/data).
- Rename it to "handwritten_digit.csv".
- The dataset is not included in this repository.
