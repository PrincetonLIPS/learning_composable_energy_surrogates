import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import CubicSpline


class FourSidedSpline(object):
    def __init__(self, cpoints):
        assert (len(cpoints) - 1) % 4 == 0
        nps = (len(cpoints) - 1) // 4  # Number of points lying along each side
        self.nps = nps
        x = np.array([i for i in range(len(cpoints))])

        self.s1 = CubicSpline(x[: nps + 1], cpoints[: nps + 1])
        self.s2 = CubicSpline(x[nps : 2 * nps + 1], cpoints[nps : 2 * nps + 1])
        self.s3 = CubicSpline(x[2 * nps : 3 * nps + 1], cpoints[2 * nps : 3 * nps + 1])
        self.s4 = CubicSpline(x[3 * nps :], cpoints[3 * nps :])

    def __call__(self, X):
        m1 = np.logical_and(X >= 0, X < self.nps)
        m2 = np.logical_and(X >= self.nps, X < 2 * self.nps)
        m3 = np.logical_and(X >= 2 * self.nps, X < 3 * self.nps)
        m4 = np.logical_and(X >= 3 * self.nps, X <= 4 * self.nps)

        Y = m1 * self.s1(X) + m2 * self.s2(X) + m3 * self.s3(X) + m4 * self.s4(X)
        return Y


def make_piecewise_spline_map(t_eval, n_cpoints):
    """
    Input:
        t_eval: vector of N eval points in [0, n + 1]
        n_cpoints: number of control points
    """
    A = []
    for i in range(n_cpoints):
        cpoints = np.zeros(n_cpoints + 1)
        cpoints[i] = 1.0
        if i == 0:
            cpoints[-1] = 1.0

        dYdc = FourSidedSpline(cpoints)(t_eval)
        A.append(dYdc)
        # pdb.set_trace()

    A = np.array(A).transpose()
    # Aij is dYi_dcj
    # so Y is np.matmul(A, c)
    assert np.all(np.isclose(A.sum(axis=1), 1.0))
    return A


if __name__ == "__main__":

    n_cpoints = 16
    t = np.linspace(0.0, float(n_cpoints) - 1e-7, 1000)
    A = make_piecewise_spline_map(t, n_cpoints)

    cpoints = np.array(
        [0.0, 0.1, 0.2, 0.5, 0.3, 1.1, 0.8, 0.6, 0.4, 0.3, 0.4, 0.8, 0.4, 0.2, 0.1, 0.1]
    )

    pdb.set_trace()

    Y = np.matmul(A, np.array(cpoints).reshape(-1, 1)).flatten()

    plt.plot(t, Y)

    plt.scatter(range(len(cpoints)), cpoints)

    plt.show()
