import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..nets.feed_forward_net import FeedForwardNet

import ray


@ray.remote
class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.p = make_p(self.args)
        self.pde = Metamaterial(self.args)
        self.fsm = FunctionSpaceMap(
            self.pde.V, self.args.data_V_dim, self.args.metamaterial_bV_dim
        )
        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)

        self.net = FeedForwardNet(args, self.fsm)

    def step(self, state_dict):
        self.net.load_state_dict(state_dict)
        self.net.eval()

        surrogate = SurrogateEnergyModel(self.args, self.net, self.fsm)

        bc, constrained_idxs, constrained_sides, constraint_mask = make_bc(
            self.args, self.fsm
        )

        force_data = make_force(self.args, self.fsm)

        params = torch.Tensor([self.args.c1, self.args.c2])

        surr_soln = surrogate.solve(params, bc, constraint_mask, force_data)

        true_soln = self.fem.solve(
            self.args,
            boundary_fn=bc,
            constrained_sides=constrained_sides,
            force_fn=force_data,
        )

        surr_soln_V = self.fsm.to_V(surr_soln)

        diff = surr_soln_V - true_soln

        error_V = fa.assemble(fa.inner(diff, diff) * self.fsm.boundary_ds)

        return error_V
