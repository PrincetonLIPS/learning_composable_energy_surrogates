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
def evaluate(args, surrogate):
    make_p(args)
    pde = Metamaterial(args)
    fsm = FunctionSpaceMap(pde.V, args.data_V_dim, args.metamaterial_bV_dim)

    fem = FenicsEnergyModel(args, pde, fsm)
    bc, _, _, constraint_mask = make_bc(args, fsm)

    force_data = make_force(args, fsm)

    params = torch.Tensor([args.c1, args.c2])

    surr_soln = surrogate.solve(params, bc, constraint_mask, force_data)

    fem.external_forces.append(fsm.to_V(force_data))
    true_soln = fem.solve(args)

    surr_soln_V = fsm.to_V(surr_soln)

    error_V = fa.assemble(fa.inner(surr_soln_V, true_soln) * fsm.boundary_ds)

    return error_V
