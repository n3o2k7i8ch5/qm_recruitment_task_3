import re

import numpy as np
import pytest

from mnist_classifier.data import get_mnist_data
from mnist_classifier.digit_classifier import DigitClassifier


class TestDigitClassifier:

    def setup_class(cls):
        cls.X, cls.y, _, _ = get_mnist_data()

    @pytest.mark.parametrize(
        "algorithm,shape,is_batch,should_raise_err",
        [
            ("cnn", (28, 28, 1), False, False),  # OK
            ("cnn", (28, 28, 1, 1), False, True),  # To many dims (1)
            ("cnn", (28, 28, 28, 784), False, True),  # To many dims (28)
            ("cnn", (6, 28, 1), False, True),  # Bad dims

            ("cnn", (1, 28, 28, 1), True, False),  # OK
            ("cnn", (2, 28, 28, 1), True, False),  # OK
            ("cnn", (3, 28, 28, 1), True, False),  # OK
            ("cnn", (1, 28, 28, 1, 1), True, True),  # To many dims
            ("cnn", (1, 1, 28, 28, 1), True, True),  # To many dims

            ("rf", (784,), False, False),  # OK
            ("rf", (784, 1), False, True),  # To many dims (1)
            ("rf", (784, 28), False, True),  # To many dims (28)
            ("rf", (28,), False, True),  # Bad dims

            ("rf", (1, 784), True, False),  # OK
            ("rf", (2, 784), True, False),  # OK
            ("rf", (3, 784), True, False),  # OK
            ("rf", (1, 784, 1), True, True),  # To many dims
            ("rf", (1, 1, 784), True, True),  # To many dims

            ("rand", (10, 10), False, False),  # OK
            ("rand", (10, 10, 1), False, True),  # To many dims (1)
            ("rand", (10, 10, 784), False, True),  # To many dims (784)
            ("rand", (10, 1000), False, True),  # Bad dims

            ("rand", (1, 10, 10), True, False),  # OK
            ("rand", (2, 10, 10), True, False),  # OK
            ("rand", (3, 10, 10), True, False),  # OK
            ("rand", (1, 10, 10, 1), True, True),  # To many dims
            ("rand", (1, 1, 10, 10), True, True),  # To many dims
        ]
    )
    def test_validate_shape(
            self,
            algorithm: str,
            shape: tuple,
            is_batch: bool,
            should_raise_err: bool
    ):
        classifier = DigitClassifier(algorithm)
        input_shape = classifier.classification.input_shape()
        X = np.ones(shape)
        if should_raise_err:
            with pytest.raises(ValueError, match=re.escape(f"Invalid shape. Must be {('N',) + input_shape} or {input_shape}.")):
                classifier.classification.validate_shape(X)
        else:
            is_batch_ = classifier.classification.validate_shape(X)
            assert is_batch == is_batch_

    @pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
    @pytest.mark.parametrize("multiple_values", [True, False])
    def test_fit_predict(self, algorithm, multiple_values):
        X = self.X if multiple_values else self.X[0, :]
        y = self.y if multiple_values else self.y[0]

        classifier = DigitClassifier(algorithm)
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        if multiple_values:
            assert y_pred.shape == self.y.shape
        else:
            assert y_pred.shape == (1,)

        assert np.all((y_pred >= 0) & (y_pred <= 9))
