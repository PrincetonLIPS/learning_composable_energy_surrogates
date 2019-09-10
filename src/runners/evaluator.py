import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..nets.feed_forward_net import FeedForwardNet

import numpy as np

import ray


@ray.remote(resources={"WorkerFlags": 0.3})
class Evaluator(object):
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        self.p = make_p(self.args)
        self.pde = Metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, cuda=False)
        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)

        self.net = FeedForwardNet(args, self.fsm)

    def step(self, state_dict):
        self.net.load_state_dict(state_dict)
        self.net.eval()

        surrogate = SurrogateEnergyModel(self.args, self.net, self.fsm)

        bc, constrained_idxs, constrained_sides, constraint_mask = make_bc(
            self.args, self.fsm
        )

        bc = bc.view(1, -1)
        bc_V = self.fsm.to_V(bc)

        n_anneal = 1 + np.random.randint(self.args.anneal_steps)

        constraint_mask = constraint_mask.unsqueeze(0)

        force_data = make_force(self.args, self.fsm).unsqueeze(0)

        params = torch.Tensor([[self.args.c1, self.args.c2]])

        factor = float(n_anneal) / self.args.anneal_steps
        surr_soln = surrogate.solve(
            params, bc * factor, constraint_mask, force_data * factor
        )

        initial_guess = np.zeros_like(bc_V.vector())
        for i in range(n_anneal):
            true_soln = self.fem.solve(
                self.args,
                boundary_fn=bc * float(i + 1) / self.args.anneal_steps,
                constrained_sides=constrained_sides,
                force_fn=force_data * float(i + 1) / self.args.anneal_steps,
                initial_guess=initial_guess,
            )
            initial_guess = true_soln.vector()[:]

        surr_soln_V = self.fsm.to_V(surr_soln)

        diff = surr_soln_V - true_soln

        error_V = fa.assemble(fa.inner(diff, diff) * self.fsm.boundary_ds)

        return error_V
