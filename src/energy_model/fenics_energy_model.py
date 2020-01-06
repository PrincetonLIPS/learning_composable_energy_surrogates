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

    def f(self, boundary_fn, initial_guess=None, return_u=False, args=None):
        if args is None:
            args = self.args
        boundary_fn = self.fsm.to_V(boundary_fn)
        self.check_initial_guess(initial_guess, boundary_fn)
        solution = self.pde.solve_problem(
            args=args, boundary_fn=boundary_fn, initial_guess=initial_guess
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

    def f_J_H(self, boundary_fn, initial_guess=None,
              return_u=False, args=None):
        if hasattr(boundary_fn, "shape") and len(boundary_fn.shape) == 2:
            raise Exception()

        boundary_fn = self.fsm.to_V(boundary_fn)
        energy, jac, solution = self.f_J(
            boundary_fn, initial_guess, return_u=True, args=args
        )
        n = self.fsm.vector_dim
        hvps = []
        for i in range(n):
            if self.args.verbose:
                print("Hessian direction {}/{}".format(i, n))
            direction = torch.zeros(self.fsm.vector_dim)
            direction[i] = 1.0
            direction = self.fsm.to_V(direction)
            hvp = fa.compute_hessian(energy, fa.Control(boundary_fn),
                                     direction)
            hvps.append(self.fsm.to_torch(
                self.fsm.V_gradient_to_ring(hvp)))
        hess = torch.stack(hvps, dim=0)

        if return_u:
            return energy, jac, hess, solution
        else:
            return energy, jac, hess

    def solve(
        self,
        args=None,
        initial_guess=None,
        boundary_fn=None,
        constrained_sides=[True, True, True, True],
        force_fn=None,
    ):
        if args is None:
            args = self.args

        keys = ["bottom", "right", "top", "left"]
        boundary_fn_dic = {}
        if boundary_fn is not None:
            boundary_fn = self.fsm.to_V(boundary_fn)
            for i, s in enumerate(constrained_sides):
                if s:
                    boundary_fn_dic[keys[i]] = boundary_fn

        external_work = None

        if force_fn is not None:
            force_fn = self.fsm.to_V(force_fn)

            def external_work(u):
                return fa.inner(u, force_fn)

        self.check_initial_guess(initial_guess, boundary_fn)
        return self.pde.solve_problem(
            args=args,
            boundary_fn_dic=boundary_fn_dic,
            external_work_fn=external_work,
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
        if not init_energy < 1e3 or not np.isfinite(init_energy):
            raise Exception(
                "Initial guess energy {} is too damn high".format(init_energy)
            )


# Taylor test
if __name__ == "__main__":
    from ..arguments import parser
    from ..pde.metamaterial import Metamaterial
    from ..maps.function_space_map import FunctionSpaceMap
    from ..data.sample_params import make_bc

    print("Preparing for Taylor test")
    args = parser.parse_args()
    pde = Metamaterial(args)
    print("Created PDE")
    fsm = FunctionSpaceMap(V=pde.V, bV_dim=args.bV_dim, cuda=False)
    print("Created FSM")
    fem = FenicsEnergyModel(args, pde, fsm)
    print("Created FEM")
    x0 = 0.1 * make_bc(args, fsm)[0]
    dx = 0.1 * make_bc(args, fsm)[0]

    fa.set_log_level(20)

    f0, J0, H0 = fem.f_J_H(x0)
    J0 = fsm.to_torch(J0)

    fa.set_log_level(30)

    for factor in [0.1 / (2 ** i) for i in range(10)]:
        # pdb.set_trace()

        delta = dx * factor
        f1 = fem.f(x0 + delta)
        fhat0 = f0
        fhat1 = fhat0 + (delta * J0).sum().data.cpu().numpy()
        fhat2 = fhat1 + torch.matmul(
            torch.matmul(delta.view(1, -1), H0),
            delta.view(-1, 1)).sum().data.cpu().numpy()/2
        first_order_remainder = np.abs(f1 - fhat0)
        second_order_remainder = np.abs(f1 - fhat1)
        third_order_remainder = np.abs(f1 - fhat2)
        # pdb.set_trace()
        print(
            "Factor: {}, first-ord remainder: {}, "
            "second-ord rem: {}, third-ord rem: {}".format(
                factor, first_order_remainder, second_order_remainder,
                third_order_remainder
            )
        )
