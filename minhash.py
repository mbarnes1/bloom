import numpy as np
import mmh3


class MinHash:
    def __init__(self, hash_count):
        self.hash_count = hash_count
        self.vec = np.zeros(self.hash_count)

    def hash(self, document):
        for seed in xrange(self.hash_count):
            minh = min([mmh3.hash64(string, seed)[0] for string in document])
            self.vec[seed] = minh
