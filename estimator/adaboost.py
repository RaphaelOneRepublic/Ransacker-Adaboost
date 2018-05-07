from estimator.model import Model
from estimator.binary_classifier import SimpleBinaryClassifier
from math import log, e, exp

sample_data = [(80, 144, +1), (93, 232, +1), (136, 275, -1),
               (147, 131, -1), (159, 69, +1), (214, 31, +1),
               (214, 152, -1), (257, 83, +1), (307, 62, -1),
               (307, 231, -1)]


class AdaptiveBooster(Model):
    def __init__(self,
                 features,
                 targets,
                 iterations=5,
                 basic_model=SimpleBinaryClassifier):
        super().__init__(features, targets)
        self.iterations = iterations
        self.basic_model = basic_model

    def fit(self):
        params = [[], []]
        weights = [1 / len(self.features)] * len(self.features)
        for i in range(self.iterations):
            simple = self.basic_model(self.features, self.targets, weights=weights)
            simple.fit()
            if simple.error > 0.5:
                break
            alpha = 0.5 * log(((1 - simple.error) / simple.error), e)
            weights = [w * exp(- alpha * t * simple.predict(f)) for w, f, t in
                       zip(weights, self.features, self.targets)]
            weights = [w / sum(weights) for w in weights]
            params[0].append(alpha)
            params[1].append(simple)
        self.predict = lambda x: sum((w * s.predict(x)) for w, s in zip(params[0], params[1]))

    def eval(self, features, targets):
        pass


if __name__ == '__main__':
    cls = AdaptiveBooster([(x[0], x[1]) for x in sample_data], [x[2] for x in sample_data])
    cls.fit()
    print(cls.targets)
    print([-1 if cls.predict(i) < 0 else 1 for i in cls.features])
