# neuroadapt/preprocessing/scaler.py

class OnlineScaler:
    """
    Simple online normalization helper.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def transform(self, value):
        if self.min is None:
            self.min = value
            self.max = value
        else:
            self.min = min(self.min, value)
            self.max = max(self.max, value)

        if self.max == self.min:
            return 0.0

        return (value - self.min) / (self.max - self.min)
