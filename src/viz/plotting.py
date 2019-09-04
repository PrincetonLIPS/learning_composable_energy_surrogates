from .. import fa_combined as fa

import numpy as np

import torch

import matplotlib.pyplot as plt


def points_along_unit_square(n):
    points = []
    N = int(n / 4)  # Points per side
    for i in range(n):
        if i < N:
            points.append((0., float(i) / N))

        elif i < 2 * N:
            points.append((float(i - N) / N, 1.0))

        elif i < 3 * N:
            points.append((1.0, 1.0 - float(i - 2 * N) / N))

        else:
            points.append((1.0 - float(i - 3 * N) / N, 0.0))

    return np.array(points)


def plot_boundary(displacement_fn, n, ax=None, normalizer=None, **kwargs):
    # Points in reference geometry
    xs = points_along_unit_square(n)
    # Corresponding displacements
    us = [displacement_fn(x) for x in xs]
    if normalizer is not None:
        norms = [np.linalg.norm(u) for u in us]
        us = np.array(us) / normalizer

    # Points in final geometry
    Xs = [(x[0] + u[0], x[1] + u[1]) for x, u in zip(xs, us)]

    Xs.append(Xs[0])

    if normalizer is not None and 'label' in kwargs:
        kwargs['label'] = kwargs['label'] + ', mean_norm: {:.3e}'.format(
            np.mean(norms))

    if ax is None:
        plot = plt.plot
    else:
        plot = ax.plot
    plot([X[0] for X in Xs], [X[1] for X in Xs], **kwargs)


def plot_vectors(locs, vecs, ax=None, normalizer=None, **kwargs):
    X = locs[:, 0]
    Y = locs[:, 1]
    U = vecs[:, 0]
    V = vecs[:, 1]

    if normalizer is not None:
        norms = [np.linalg.norm(vec) for vec in vecs]
        vecs = np.array(vecs) / normalizer

    if ax is None:
        ax = plt

    if normalizer is not None and 'label' in kwargs:
        kwargs['label'] = kwargs['label'] + ', mean_norm: {:.3e}'.format(
            np.mean(norms))

    qplot = ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', **kwargs)
    qk = ax.quiverkey(qplot,
                      0.9,
                      0.9,
                      2,
                      r'\frac{dE}{du}',
                      labelpos='E',
                      coordinates='figure')
    ax.scatter(X, Y, color='k')
