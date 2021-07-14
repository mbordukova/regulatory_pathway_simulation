import numpy as np


class Sampler:
    def __init__(self, random_state: np.random.RandomState = np.random.RandomState(seed=0)):
        self.random_state = random_state

    def get_bernoulli_sample(self, bernoulli_parameter: float):
        return self.random_state.binomial(n=1, p=bernoulli_parameter, size=1)
