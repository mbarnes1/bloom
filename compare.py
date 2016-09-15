from bloomfilter import BloomFilter
from minhash import MinHash
import numpy as np
from itertools import izip
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt


def main():
    m = 1000000  # max hash value
    h = 2000  # number of hash functions
    jaccard = 0.8
    N = np.linspace(10, 10**3, num=10).astype('int')

    jaccard_minhash = []
    jaccard_bloom = []
    jaccard_true = []

    for n in N:
        d1 = set([str(x) for x in range(n)])
        min_d2 = int(n*(1.-jaccard)/(1. + jaccard))
        d2 = set([str(x) for x in range(min_d2, min_d2 + n)])

        b1 = BloomFilter(m, h)
        b2 = BloomFilter(m, h)

        mh1 = MinHash(h)
        mh2 = MinHash(h)

        for s1, s2 in izip(d1, d2):
            b1.add(s1)
            b2.add(s2)
        mh1.hash(d1)
        mh2.hash(d2)

        jaccard_minhash.append(1.-hamming(mh1.vec, mh2.vec))
        jaccard_bloom.append(1-2*float(sum(np.not_equal(b1.bit_array, b2.bit_array)))/(sum(b1.bit_array) + sum(b2.bit_array)))
        jaccard_true.append(float(len(d1.intersection(d2)))/len(d1.union(d2)))

    plt.plot(N, np.array([jaccard_bloom, jaccard_minhash, jaccard_true]).T)
    plt.legend(['Bloom Filter', 'MinHash', 'True'], loc='upper left')
    plt.xlabel('Number of strings')
    plt.ylabel('Jaccard Coefficient')
    plt.title('Jaccard Approximation Through Hashing')
    plt.show()

if __name__ == '__main__':
    main()