import numpy as np


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
    predict = None

    def __init__(self, features, targets=None):
        """
        Constructor
        Args:
            features(list): A list of features
            targets(list) optional: A list of target values, if None, then unsupervised learning
        """
        self.features = features
        self.targets = targets

    def fit(self):
        """Fit the input data"""
        raise NotImplementedError

    def eval(self, features, targets):
        """Evaluate the estimated model using the input test features and targets"""
        raise NotImplementedError

