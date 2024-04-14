import numpy as np
import math


dataset = np.load("./datasets/mnist.npy")
x, y = np.split(dataset, np.array([784, ]), axis=1)
x = x.astype(np.float32) / 255.0
del dataset
x = np.expand_dims(np.reshape(x, (-1, 28, 28)), axis=-1)
y = np.squeeze(y)
x_train, y_train = x[:60000], y[:60000]
x_test, y_test = x[60000:], y[60000:]
del x, y


def mnist_training_dataset(batch_size: int = 128):
    num_samples = x_train.shape[0]
    num_batches = math.ceil(num_samples / batch_size)
    start_idx = 0
    end_idx = batch_size
    for i in range(num_batches):
        x_batch = x_train[start_idx: end_idx]
        y_batch = y_train[start_idx: end_idx]
        start_idx += batch_size
        end_idx = min(end_idx + batch_size, num_samples)
        yield x_batch, y_batch
