from estimator.model import Model
import numpy as np
from matplotlib import pyplot as plt


class ScalarLinearRegressor(Model):
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
        self.predict = lambda _x: self.params[0] + self.params[1] * _x
        self.error = 0
        for feature, target in zip(self.features, self.targets):
            self.error += abs(self.predict(feature) - target)

    def eval(self, features, targets):
        return [targets[i] - self.predict(features[i]) for i in range(len(features))]

    def display(self):
        """Display the estimated model"""
        plt.figure()
        x = np.array(self.features)
        y = np.array(self.targets)
        plt.scatter(x, y)
        ScalarLinearRegressor.abline(self.params[1], self.params[0])
        plt.show()

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
