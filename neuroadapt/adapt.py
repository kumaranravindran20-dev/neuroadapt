class OnlineAdaptor:
    """
    Online learning rule inspired by brain plasticity.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def compute_error(self, prediction, target):
        return target - prediction

    def adapt(self, layer, error):
        layer.weight += self.learning_rate * error
