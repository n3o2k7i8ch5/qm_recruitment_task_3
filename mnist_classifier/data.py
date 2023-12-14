import numpy as np
import torchvision.datasets as datasets


def get_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    X_train = mnist_train.data
    X_train = X_train.resize(*X_train.shape, 1).numpy()
    y_train = mnist_train.targets.numpy()

    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    X_test = mnist_test.data
    X_test = X_test.resize(*X_test.shape, 1).numpy()
    y_test = mnist_test.targets.numpy()

    return X_train, y_train, X_test, y_test
