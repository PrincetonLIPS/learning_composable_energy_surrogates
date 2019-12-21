
import copy
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial, make_metamaterial_mesh
import numpy as np


class ComposedFenicsEnergyModel(object):
    def __init__(self, args, n_high, n_wide, c1s, c2s):
        assert n_high == n_wide
        assert len(c1s) == n_high * n_wide
        assert len(c2s) == n_high * n_wide
        self.args = copy.deepcopy(args)
        del args
        if self.args.L0 is None:
            self.args.L0 = 1./self.args.n_cells

        c1s = np.repeat(
                np.repeat(np.reshape(c1s, (n_high, n_wide)),
                          self.args.n_cells, axis=1),
                self.args.n_cells, axis=0)
        c2s = np.repeat(
                np.repeat(np.reshape(c2s, (n_high, n_wide)),
                          self.args.n_cells, axis=1),
                self.args.n_cells, axis=0)

        self.args.n_cells *= n_high
        mesh = make_metamaterial_mesh(
            self.args.L0, c1s, c2s, self.args.pore_radial_resolution,
            self.args.min_feature_size, self.args.metamaterial_mesh_size,
            self.args.n_cells, self.args.porosity)

        self.pde = Metamaterial(self.args, mesh)

    def solve(
        self,
        args=None,
        initial_guess=None,
        boundary_fn=None,
        constrained_sides=[True, True, True, True],
        force_fn=None,
    ):
        """
        params:           [n_cells, n_params] tensor
        boundary_data:    [n_global_locs, geometric_dim] tensor
        """
        if args is None:
            args = self.args

        keys = ["bottom", "right", "top", "left"]
        boundary_fn_dic = {}
        if boundary_fn is not None:
            boundary_fn = fa.interpolate(boundary_fn, self.pde.V)
            for i, s in enumerate(constrained_sides):
                if s:
                    boundary_fn_dic[keys[i]] = boundary_fn

        external_work = None
        if force_fn is not None:
            force_fn = fa.interpolate(force_fn, self.pde.V)

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
        init_guess_fn = fa.Function(self.pde.V)
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

if __name__ == '__main__':
    from ..arguments import parser
    from ..pde.metamaterial import Metamaterial
    from ..maps.function_space_map import FunctionSpaceMap
    from .surrogate_energy_model import SurrogateEnergyModel
    from ..data.sample_params import make_bc
    from ..data.random_fourier_fn import make_random_fourier_expression
    import pdb

    args = parser.parse_args()

    np.random.seed(0)

    fa.set_log_level(20)

    n_high = 2
    n_wide = 2
    c1s = np.random.randn(2*2) * 0.01
    c2s = np.random.randn(2*2) * 0.01

    cfem = ComposedFenicsEnergyModel(args, n_high, n_wide, c1s, c2s)

    boundary_fn = fa.Constant((0.0, 0.0))

    initial_guess = fa.Function(cfem.pde.V)
    initial_guess.vector().set_local(
        np.random.randn(len(initial_guess.vector()))*0.00001)

    cfem.solve(boundary_fn=boundary_fn,
               initial_guess=initial_guess.vector())
    pdb.set_trace()
