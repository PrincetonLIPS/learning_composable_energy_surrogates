"""EnergyModel which solves Fenics pde. Ground truth or simple-mesh approx."""

from .. import fa_combined as fa
import pdb
import torch
import numpy as np


class FenicsEnergyModel(object):
    def __init__(self, args, pde, function_space_map):
        self.args = args
        self.pde = pde
        self.fsm = function_space_map
        self.boundary_constraints = {}
        self.external_forces = []

    def external_work_fn(self, u):
        work = fa.Constant(0.0) * fa.inner(u, u)
        for wfunc in self.external_forces:
            work = work + wfunc(u)
        return work

    def f(self, boundary_fn, initial_guess=None, return_u=False, args=None):
        if args is None:
            args = self.args
        boundary_fn = self.fsm.to_V(boundary_fn)
        self.check_initial_guess(initial_guess, boundary_fn)
        solution = self.pde.solve_problem(
            args=args,
            boundary_fn=boundary_fn,
            external_work_fn=self.external_work_fn,
            initial_guess=initial_guess,
        )
        energy = self.pde.energy(solution)
        if return_u:
            return energy, solution
        else:
            return energy

    def f_J(self, boundary_fn, initial_guess=None, return_u=False, args=None):
        if hasattr(boundary_fn, "shape") and len(boundary_fn.shape) == 2:
            raise Exception()
        boundary_fn = self.fsm.to_V(boundary_fn)
        energy, solution = FenicsEnergyModel.f(
            self, boundary_fn, initial_guess, return_u=True, args=args
        )
        jac = fa.compute_gradient(energy, fa.Control(boundary_fn))
        # pdb.set_trace()
        # jac = self.fsm.to_dV(jac)
        # solution = self.fsm.to_dV(solution)
        if return_u:
            return energy, jac, solution
        else:
            return energy, jac

    def f_J_Hvp(
        self, boundary_fn, vector, initial_guess=None, return_u=False, args=None
    ):
        if hasattr(boundary_fn, "shape") and len(boundary_fn.shape) == 2:
            raise Exception()

        boundary_fn = self.fsm.to_V(boundary_fn)
        energy, jac, solution = self.f_J(
            boundary_fn, initial_guess, return_u=True, args=args
        )
        direction = self.fsm.to_V(vector)
        hvp = fa.compute_hessian(energy, fa.Control(boundary_fn), direction)
        # pdb.set_trace()
        # jac = self.fsm.to_dV(jac)
        # hvp = self.fsm.to_dV(hvp)
        # solution = self.fsm.to_dV(solution)

        if return_u:
            return energy, jac, hvp, solution
        else:
            return energy, jac, hvp

    def solve(self, args=None, initial_guess=None):
        if args is None:
            args = self.args
        self.check_initial_guess(initial_guess)
        return self.pde.solve_problem(
            args=args,
            boundary_fn_dic=self.boundary_constraints,
            external_work_fn=self.external_work_fn,
            initial_guess=initial_guess,
        )

    def check_initial_guess(self, initial_guess=None, boundary_fn=None):
        init_guess_fn = fa.Function(self.fsm.V)
        if initial_guess is not None:
            init_guess_fn.vector().set_local(initial_guess)
        if boundary_fn is not None:
            bc = fa.DirichletBC(self.pde.V, boundary_fn, self.pde.exterior)
            bc.apply(init_guess_fn.vector())
        init_energy = self.pde.energy(init_guess_fn)
        if self.args.verbose:
            print("Init energy: {}".format(init_energy))
        if not init_energy < 1e2 or not np.isfinite(init_energy):
            raise Exception(
                "Initial guess energy {} is too damn high".format(init_energy)
            )

    def clear_boundary(self):
        self.boundary_constraints = {}

    def clear_external(self):
        self.external_forces = []

    def apply_boundary(self, side, bc):
        assert side in ["left", "right", "bottom", "top"]
        self.boundary_constraints[side] = bc

    def apply_force(self, force_fn):
        self.clear_external()
        self.external_forces.append(force_fn)

    def apply_linear(self, linear_fn):
        self.apply_force(linear_fn)

    def apply_quadratic(self, quad_fn):
        self.apply_force(quad_fn)


class SurrogateFenicsEnergyModel(FenicsEnergyModel):
    """Hack to give FenicsEnergyModel the same f_J_Hvp syntax as the others."""

    def f(self, boundary_fn, params):
        return super(SurrogateFenicsEnergyModel, self).f(boundary_fn)

    def f_J(self, boundary_fn, params):
        return super(SurrogateFenicsEnergyModel, self).f_J(boundary_fn)

    def f_J_Hvp(self, boundary_fn, params, vector):
        return super(SurrogateFenicsEnergyModel, self).f_J_Hvp(boundary_fn, vector)
