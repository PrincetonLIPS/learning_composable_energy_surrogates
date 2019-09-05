import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..data.example import Example
import random
import numpy as np
import ray


@ray.remote
class Evaluator(object):
    def __init__(self, args):
        self.args = args

    def step(self, surrogate):
        make_p(self.args)
        pde = Metamaterial(self.args)
        fsm = FunctionSpaceMap(pde.V, self.args.data_V_dim,
                               self.args.metamaterial_bV_dim)

        fem = FenicsEnergyModel(self.args, pde, fsm)
        bc, _, _, constraint_mask = make_bc(self.args, fsm)

        force_data = make_force(self.args, fsm)

        params = torch.Tensor([self.args.c1, self.args.c2])

        surr_soln = surrogate.solve(params, bc, constraint_mask, force_data)

        fem.external_forces.append(fsm.to_V(force_data))
        true_soln = fem.solve(self.args)

        surr_soln_V = fsm.to_V(surr_soln)

        error_V = fa.assemble(fa.inner(surr_soln_V, true_soln) * fsm.boundary_ds)

        return error_V
