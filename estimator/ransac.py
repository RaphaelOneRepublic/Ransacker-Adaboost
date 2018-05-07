from estimator.model import Model
from estimator.scalar_linear import ScalarLinearRegressor
from random import shuffle
from matplotlib import pyplot as plt


class Ransacker(Model):
    """
    The ransac model estimator
    """
    sample_data = [(-2, 0), (0, 0.9), (2, 2.0),
                   (3, 6.5), (4, 2.9), (5, 8.8),
                   (6, 3.95), (8, 5.03), (10, 5.97),
                   (12, 7.1), (13, 1.2), (14, 8.2),
                   (16, 8.5), (18, 10.1)
                   ]

    inlier_features = None
    inlier_targets = None

    def __init__(self,
                 features,
                 targets,
                 basic_model=ScalarLinearRegressor,
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
        if self.features is None:
            raise ValueError("Features uninitialized")
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
                    self.predict = basic.predict

        if self.params is None:
            raise ValueError("Ransac failed")

    def eval(self, features, targets):
        return [targets[i] - self.predict(features[i]) for i in range(len(features))]

    def display(self):
        plt.figure()
        plt.scatter(self.features, self.targets, c="r")
        plt.scatter(self.inlier_features, self.inlier_targets, c="b")
        ScalarLinearRegressor.abline(self.params[1], self.params[0], color="yellow")
        plt.show()


if __name__ == '__main__':
    ransac = Ransacker([i[0] for i in Ransacker.sample_data], [i[1] for i in Ransacker.sample_data], iterations=10)
    ransac.fit()
    ransac.display()
