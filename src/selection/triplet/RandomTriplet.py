import numpy as np
from skquery.strategy import QueryStrategy


class RandomTriplet(QueryStrategy):
    def __init__(self):
        super().__init__()

    def fit(self, X, oracle, partition=None):
        triplet = []
        constraints = {"triplet": triplet}
        candidates = [np.random.choice(range(X.shape[0]), size=3, replace=False).tolist() for _ in range(oracle.budget)]

        for a, p, n in candidates:
            left_hand = oracle.query(a, p)
            if left_hand:
                triplet.append((a, p, n))
            else:
                triplet.append((a, n, p))

        return constraints
