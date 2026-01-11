class Neuron:
    """
    Basic neuron with a weight.
    """
    def __init__(self, weight=0.5):
        self.weight = weight

    def activate(self, x):
        return x * self.weight


class AdaptiveLayer:
    """
    Adaptive layer that updates its weight based on error.
    """
    def __init__(self, learning_rate=0.01):
        self.weight = 0.5
        self.learning_rate = learning_rate

    def forward(self, x):
        return x * self.weight

    def update(self, error):
        self.weight += self.learning_rate * error
