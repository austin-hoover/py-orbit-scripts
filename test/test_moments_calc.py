"""Make sure ORBIT-calculated second-order moments are correct."""
from __future__ import print_function
from pprint import pprint

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis


# Choose mean and covariance.
np.random.seed(0)
decimals = 3

cov = np.identity(6)
for (i, j) in [(0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (2, 5)]:
    cov[i, j] = cov[j, i] = 0.9 * np.random.uniform(-1.0, 1.0)
cov = 100.0 * np.around(cov, decimals)

mean = np.zeros(6)
for i in [0, 3, 5]:
    mean[i] = np.random.uniform(3)
mean = np.around(mean, decimals)

print("cov theory:")
print(cov)
print("mean theory:")
print(mean)

# Generate data.
X = np.random.multivariate_normal(mean, cov, size=100000)
print("cov numpy:")
np_cov = np.cov(X.T)
print(np.around(np_cov, decimals=3))
print("mean numpy:")
print(np.around(np.mean(X, axis=0), decimals))

# Recompute mean and covariance from bunch.
bunch = Bunch()
for (x, xp, y, yp, z, dE) in X:
    bunch.addParticle(x, xp, y, yp, z, dE)
bunch_twiss_analysis = BunchTwissAnalysis()
bunch_twiss_analysis.analyzeBunch(bunch)

mean = np.array([bunch_twiss_analysis.getAverage(i) for i in range(6)])
print("mean orbit:")
print(np.around(mean, decimals))

order = 2
dispersion_flag = 0
emit_norm_flag = 0
bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
cov = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        cov[i, j] = bunch_twiss_analysis.getCorrelation(i, j)
print("cov orbit:")
pprint(np.around(cov, decimals))

print("numpy | orbit | absolute difference")
dims = ["x", "xp", "y", "yp", "z", "dE"]
for i in range(6):
    for j in range(6):
        print(
            "{}-{}: {:.3f} | {:.3f} | {:.3e}".format(
                dims[i],
                dims[j],
                np_cov[i, j],
                cov[i, j],
                np.abs(np_cov[i, j] - cov[i, j]),
            )
        )