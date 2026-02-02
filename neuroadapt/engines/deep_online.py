# neuroadapt/engines/deep_online.py

class DeepOnlineEngine:
    """
    Engine to manage online updates for complex models.
    """

    def update(self, model, x, y):
        model.adapt(x, y)
        return model
