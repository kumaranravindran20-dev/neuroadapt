from neuroadapt.core import AdaptiveLayer
from neuroadapt.adapt import OnlineAdaptor


class AdaptiveModel:
    """
    Brain-inspired adaptive model with continuous learning.
    """
    def __init__(self, learning_rate=0.01):
        self.layer = AdaptiveLayer(learning_rate)
        self.adaptor = OnlineAdaptor(learning_rate)

    def predict(self, x):
        return self.layer.forward(x)

    def adapt(self, x, target):
        prediction = self.predict(x)
        error = self.adaptor.compute_error(prediction, target)
        self.adaptor.adapt(self.layer, error)
        return self  # enables chaining
