# neuroadapt/models/adaptive_lstm.py

import numpy as np

class AdaptiveLSTMModel:
    """
    Experimental adaptive LSTM-style model.
    Uses sliding window + incremental updates.
    """

    def __init__(self, input_size, window_size=5, learning_rate=0.001):
        self.input_size = input_size
        self.window_size = window_size
        self.learning_rate = learning_rate

        # simple weight placeholder (not full DL yet)
        self.weights = np.random.randn(input_size * window_size)

    def _flatten_window(self, window):
        return np.array(window).flatten()

    def predict(self, window):
        x = self._flatten_window(window)
        return np.dot(self.weights, x)

    def adapt(self, window, target):
        x = self._flatten_window(window)
        prediction = self.predict(window)
        error = target - prediction
        self.weights += self.learning_rate * error * x
        return self
