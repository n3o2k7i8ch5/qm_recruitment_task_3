import itertools
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.autograd import Variable
from torch.nn import Conv2d, Linear, Module, CrossEntropyLoss
from torch.nn.functional import relu, max_pool2d, dropout, log_softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class DigitClassificationInterface(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @abstractmethod
    def input_shape(self):
        pass

    def validate_shape(self, X: np.ndarray) -> bool:
        """
        :param X: Data which shape should be validated.
        :return: False if provided data is a single observation, True if a batch of observations.
        """
        valid_shape = self.input_shape()

        if X.shape == valid_shape:
            return False

        if X.shape[1:] == valid_shape:
            return True

        raise ValueError(f"Invalid shape. Must be {('N',) + valid_shape} or {valid_shape}.")


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=5)
        self.conv2 = Conv2d(32, 32, kernel_size=5)
        self.conv3 = Conv2d(32, 64, kernel_size=5)
        self.linear1 = Linear(3*3*64,  256)
        self.linear2 = Linear(256, 10)

    def forward(self, X: torch.Tensor):
        X = relu(self.conv1(X.reshape(-1, 1, 28, 28)))
        X = relu(max_pool2d(self.conv2(X), 2))
        X = dropout(X, p=0.5, training=self.training)
        X = relu(max_pool2d(self.conv3(X), 2))
        X = dropout(X, p=0.5, training=self.training)
        X = X.view(-1, 3*3*64)
        X = relu(self.linear1(X))
        X = dropout(X, training=self.training)
        X = self.linear2(X)
        return log_softmax(X, dim=1)


class CNNDigitClassification(DigitClassificationInterface):

    BATCH_SIZE = 2048
    EPOCHS = 1

    def __init__(self):
        self.cnn = CNN()

    def fit(self, X: np.ndarray, y: np.ndarray):

        train = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
        train_loader = DataLoader(train, batch_size=self.BATCH_SIZE, shuffle=False)

        optimizer = Adam(self.cnn.parameters())
        error = CrossEntropyLoss()

        self.cnn.train()

        for epoch, (batch, (X_batch, y_batch)) in itertools.product(range(self.EPOCHS), enumerate(train_loader)):
            X_batch = Variable(X_batch).float()
            y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = self.cnn.forward(X_batch)
            loss = error(output, y_batch)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray):
        X_torch = torch.as_tensor(X, dtype=torch.float)
        return self.cnn.forward(X_torch).argmax(dim=1).detach().numpy()

    def input_shape(self):
        return (28, 28, 1)


class RandomForestDigitClassification(DigitClassificationInterface):

    def __init__(self, **kwargs):
        self.random_forest = RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.random_forest.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.random_forest.predict(X)

    def input_shape(self):
        return (28*28,)


class RandomValueDigitClassification(DigitClassificationInterface):

    classes: list

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y).tolist()

    def predict(self, X: np.ndarray):
        return np.random.choice(self.classes, size=(len(X),))

    def input_shape(self):
        return (10, 10)
