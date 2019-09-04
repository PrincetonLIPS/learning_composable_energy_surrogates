"""Base class for Solver"""
from .. import fa_combined as fa
import copy
import numpy as np


class Solver(object):
    """Base class for Solvers.

    Manully implement Newton solver. (https://fenicsproject.org/qa/536/newton-method-programmed-manually/)
    Bugs may exsit, not sure whether it's worthwhile investigating this.
    Happy with current FEniCS solver for now.
    """
    def __init__(self, args):
        self.args = args

    def solve(self, dE, jacE, u, delta_u, bcs, solver_args, ffc_options):

        if self.args.manual_solver:

            # initial_u = fa.Expression(("0", "-x[1]/1.5*0.1"), degree=1)
            # initial_u = fa.interpolate(initial_u, V)
            # u.vector().set_local(initial_u.vector())

            for bc in bcs:
                bc.apply(u.vector())

            bcs_du = []
            for bc in bcs:
                bc.homogenize()
                bcs_du = bcs_du + [bc]

            a_tol, r_tol = 1e-7, 1e-10
            nIter = 0
            eps = 1

            while eps > 1e-10 and nIter < 1000:
                # In each iteration, one linear system gets solved.
                nIter += 1
                A, b = fa.assemble_system(jacE, -dE, bcs_du)
                fa.solve(A, delta_u.vector(), b)  # Determine step direction
                eps = np.linalg.norm(np.array(delta_u.vector()), ord=2)
                fnorm = b.norm('l2')
                lmbda = 0.1  # step size, initially 1
                u.vector()[:] += lmbda * delta_u.vector()  # New u vector
                print('{0:2d}  {1:3.2E}  {2:5e}'.format(nIter, eps, fnorm))

        else:
            fa.solve(dE == 0,
                     u,
                     bcs,
                     J=jacE,
                     solver_parameters=solver_args,
                     form_compiler_parameters=ffc_options)
