import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

raw_data = [(-2, 0), (0, 0.9), (2, 2.0),
            (3, 6.5), (4, 2.9), (5, 8.8),
            (6, 3.95), (8, 5.03), (10, 5.97),
            (12, 7.1), (13, 1.2), (14, 8.2),
            (16, 8.5), (18, 10.1)
            ]


class Model(object):
    """
    Abstract class for estimating a model function from a set of features to a target value
    """

    """The input feature list of the estimator"""
    features = None

    """The input target list of the estimator"""
    targets = None

    """Used to store the parameters of the model"""
    params = None

    """The error of the estimated model on the input target values"""
    error = np.inf

    """The estimated model, takes a feature (list) and returns a target value"""
    model = None

    def __init__(self, features, targets):
        """
        Constructor
        Args:
            features(list): A list of features
            targets(list): A list of target values
        """
        self.features = features
        self.targets = targets

    def fit(self):
        """Fit the input data"""
        raise NotImplementedError

    def eval(self, features, targets):
        """Evaluate the estimated model using the input test features and targets"""
        raise NotImplementedError

    @staticmethod
    def abline(slope, intercept, color="r"):
        """
        Plot a line from slope and intercept
        Must already have a plot figure available

        Args:
            slope(float): Slope of the line to be depicted
            intercept(float): Intercept of the line on the y-axis
            color(String) optional: Color of the line, default is red, to be distinguished from default dot color
        """
        axes = plt.gca()
        x_val = np.array(axes.get_xlim())
        y_val = intercept + slope * x_val
        plt.plot(x_val, y_val, '--', c=color)


class ScalarLinearModel(Model):
    """
    Estimator for linear models
    Features and target values must have the same shape and type
    This uses the pseudo-inverse to calculate normal equations
    """

    def fit(self):
        if len(self.features) != len(self.targets):
            raise ValueError("Feature and target shapes do not agree")

        x = np.column_stack((np.ones(len(self.features)), np.array(self.features)))
        x_inv = np.linalg.pinv(x)
        y = np.array(self.targets)
        self.params = np.matmul(x_inv, y)
        self.model = lambda _x: self.params[0] + self.params[1] * _x
        self.error = 0
        for feature, target in zip(self.features, self.targets):
            self.error += abs(self.model(feature) - target)

    def eval(self, features, targets):
        return [targets[i] - self.model(features[i]) for i in range(len(features))]

    def display(self):
        """Display the estimated model"""
        plt.figure()
        x = np.array(self.features)
        y = np.array(self.targets)
        plt.scatter(x, y)
        Model.abline(self.params[1], self.params[0])
        plt.show()


class Ransacker(Model):
    """T
    he ransac model estimator

    """
    inlier_features = None
    inlier_targets = None

    def __init__(self,
                 features,
                 targets,
                 basic_model=ScalarLinearModel,
                 min_samples=2,
                 min_inliers=0.5,
                 iterations=100,
                 eps=0.5):
        """
        Constructor
        Args:
            features(list): A list of features
            targets(list): A list of target values
            basic_model(Model) optional:
             the underlying model with which ransac works for sampled data, default is ScalarLinearModel
            min_samples(int) optional: minimum samples required to perform fitting,
            min_inliers(float) optional: minimum percentage of features that need to have consensus
            iterations
            eps(float) optional: threshold
        """
        super().__init__(features, targets)
        self.basic_model = basic_model
        self.min_samples = min_samples
        self.min_inliers = min_inliers
        self.iterations = iterations
        self.eps = eps

    def fit(self):
        if len(self.features) != len(self.targets):
            raise ValueError("Feature and target shapes do not agree")
        if len(self.features) < self.min_samples:
            raise ValueError("Not enough features")
        basic = self.basic_model([], [])
        for i in range(self.iterations):
            indices = list(range(len(self.features)))
            shuffle(indices)
            basic.features = [self.features[i] for i in indices[:self.min_samples]]
            basic.targets = [self.targets[i] for i in indices[:self.min_samples]]
            feature_test = [self.features[i] for i in indices[self.min_samples:]]
            target_test = [self.targets[i] for i in indices[self.min_samples:]]
            basic.fit()

            errors = basic.eval(feature_test, target_test)

            for index, inlier in enumerate([abs(i) < self.eps for i in errors]):
                if inlier:
                    basic.features.append(feature_test[index])
                    basic.targets.append(target_test[index])
            if len(basic.features) >= self.min_inliers * len(self.features):
                basic.fit()
                if basic.error < self.error:
                    self.params = basic.params
                    self.inlier_features = basic.features
                    self.inlier_targets = basic.targets
                    self.error = basic.error
                    self.model = basic.model

        if self.params is None:
            raise ValueError("Ransac failed")

    def eval(self, features, targets):
        return [targets[i] - self.model(features[i]) for i in range(len(features))]

    def display(self):
        plt.figure()
        plt.scatter(self.features, self.targets, c="r")
        plt.scatter(self.inlier_features, self.inlier_targets, c="b")
        Model.abline(self.params[1], self.params[0], color="yellow")
        plt.show()


if __name__ == '__main__':
    ransac = Ransacker([i[0] for i in raw_data], [i[1] for i in raw_data], iterations=10)
    ransac.fit()
    ransac.display()
