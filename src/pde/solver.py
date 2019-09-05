"""Base class for Solver"""
from .. import fa_combined as fa
import copy
import numpy as np


class Solver(object):
    """Base class for Solvers.

    Manully implement Newton solver.
    (https://fenicsproject.org/qa/536/newton-method-programmed-manually/)
    Bugs may exsit, not sure whether it's worthwhile investigating this.
    Happy with current FEniCS solver for now.
    """

    def __init__(self, args):
        self.args = args

    def solve(self, dE, jacE, u, delta_u, bcs, solver_args, ffc_options):
        fa.solve(
            dE == 0,
            u,
            bcs,
            J=jacE,
            solver_parameters=solver_args,
            form_compiler_parameters=ffc_options,
        )
