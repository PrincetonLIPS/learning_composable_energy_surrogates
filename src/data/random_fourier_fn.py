from .. import fa_combined as fa
import numpy as np
import math


def get_random_fourier_fn(N, amp_scale, freq_scale):
    """Create a random fourier fn
    Random fourier function a la Rahimi and Recht
    See the Rahimi and Recht papers for impl. details
    I *think* this approximates function draws from a GP prior
    with kernel variance freq_scale
    and concentration/scale of the GP prior controlled by amp_scale
    where the approximation becomes exact as N -> infty """
    alpha = np.random.randn(1, 1, N)
    wx = np.random.randn(1, 1, N)
    wy = np.random.randn(1, 1, N)
    b = 2 * math.pi * np.random.randn(1, 1, N)

    def z_func(x, y):
        # returns the fn at some set of locations indexed with X and Y
        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        fourier = np.cos(freq_scale * (wx * x + wy * y) + b)
        zs = amp_scale * np.sum(alpha * fourier, axis=2) / np.sqrt(N)
        return zs

    return z_func


def make_random_fourier_expression(dim, N, amp_scale, freq_scale, element):
    """Make a random fourier expression with given dim and fourier params.

    Output is a Fenics UserExpression,
    ready for interpolation to a FunctionSpace."""

    u_fns = []
    # Create an independant fourier fn for each dim
    for i in range(dim):
        u_fns.append(get_random_fourier_fn(N, amp_scale, freq_scale))

    class RandomFourierExpression(fa.UserExpression):
        def eval(self, value, x):
            for i in range(dim):
                value[i] = sum(u_fns[i]([[x[0]]], [[x[1]]]))

        def value_shape(self):
            return (len(u_fns), )

    return RandomFourierExpression(element=element)
