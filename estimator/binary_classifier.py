from estimator.model import Model
import numpy as np


class SimpleBinaryClassifier(Model):
    weights = None

    def __init__(self, features, targets, weights=None):
        super().__init__(features, targets)
        if weights is None:
            weights = [1 / len(features)] * len(features)
        self.weights = weights
        self.x_weight = sorted(zip([feature[0] for feature in features], targets, weights), key=lambda x: x[0])
        self.y_weight = sorted(zip([feature[1] for feature in features], targets, weights), key=lambda x: x[0])

    def fit(self):
        on_x = True
        threshold = None
        positive = True
        self.error = np.inf
        for feature in self.features:
            x_loss, x_positive = self.loss(x=feature[0])
            y_loss, y_positive = self.loss(y=feature[1])
            if self.error > x_loss:
                self.error = x_loss
                positive = x_positive
                on_x = True
                threshold = feature[0]
            if self.error > y_loss:
                self.error = y_loss
                positive = y_positive
                on_x = False
                threshold = feature[1]
        if on_x and positive:
            self.predict = lambda f: -1 if f[0] > threshold else 1
        if on_x and not positive:
            self.predict = lambda f: 1 if f[0] > threshold else -1
        if not on_x and positive:
            self.predict = lambda f: -1 if f[1] > threshold else 1
        if not on_x and not positive:
            self.predict = lambda f: 1 if f[1] > threshold else -1

    def loss(self, x=None, y=None):
        """
        Up 0
        Left 0
        :param x:
        :param y:
        :return:
        """
        loss = 0
        positive = False
        if (x is None and y is None) or (x is not None and y is not None):
            raise ValueError("Must specify either x or y")
        if x is not None:
            pre = -1
            for xx, target, weight in self.x_weight:
                if xx > x:
                    pre = 1
                if target != pre:
                    loss += weight
        if y is not None:
            pre = -1
            for yy, target, weight in self.y_weight:
                if yy > y:
                    pre = 1
                if target != pre:
                    loss += weight
        if loss > 0.5:
            loss = 1 - loss
            positive = True
        return loss, positive

    def eval(self, features, targets):
        pass
