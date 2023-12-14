import numpy as np

from mnist_classifier.digit_classifications import CNNDigitClassification, \
    RandomForestDigitClassification, RandomValueDigitClassification


InvalidMethodError = ValueError("Invalid method")


class DigitClassifier:

    def __init__(self, algorithm: str, model_params: dict = None):
        self.algorithm = algorithm
        self.model_params = model_params or {}

        if algorithm == "cnn":
            self.classification = CNNDigitClassification()
        elif algorithm == "rf":
            self.classification = RandomForestDigitClassification(**self.model_params)
        elif algorithm == "rand":
            self.classification = RandomValueDigitClassification()
        else:
            raise InvalidMethodError

    @staticmethod
    def is_batch(X: np.ndarray) -> bool:
        if X.shape == (28, 28, 1):
            return False

        if X.shape[1:] == (28, 28, 1):
            return True

        raise ValueError("Invalid shape. Must be (28, 28, 1) or (N, 28, 28, 1).")

    def preprocess_data(self, X: np.ndarray) -> (np.ndarray, bool):
        is_batch = self.is_batch(X)

        if self.algorithm == "cnn":
            return X.reshape(-1, 1, 28, 28), is_batch
        elif self.algorithm == "rf":
            return X.reshape(-1, 28*28), is_batch
        elif self.algorithm == "rand":
            return X.reshape(-1, 28, 28)[:, 9:19, 9:19], is_batch
        else:
            raise InvalidMethodError

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, is_batch = self.preprocess_data(X)
        if not is_batch:
            y = y.reshape((1,))
        return self.classification.fit(X, y)

    def predict(self, X: np.ndarray):
        X, _ = self.preprocess_data(X)
        return self.classification.predict(X)
