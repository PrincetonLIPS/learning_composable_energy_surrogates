import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_compression_deploy_bc
from ..data.example import Example
from ..nets.feed_forward_net import FeedForwardNet
from ..energy_model.composed_energy_model import ComposedEnergyModel
from ..energy_model.composed_fenics_energy_model import ComposedFenicsEnergyModel
from ..util.solve import solve

import random
import numpy as np
import ray

from .collector import CollectorBase


class DeployCollectorBase(CollectorBase):
    def __init__(self, args, seed, state_dict):
        self.args = args
        np.random.seed(seed)
        make_p(args)
        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.fem = FenicsEnergyModel(args, self.pde, self.fsm)
        net = FeedForwardNet(args, self.fsm)
        net.load_state_dict(state_dict)
        sem = SurrogateEnergyModel(args, net, self.fsm)
        cem = ComposedEnergyModel(args, sem,
                                  args.n_high, args.n_wide)
        cem_boundary, constraint_mask = make_compression_deploy_bc(args, cem)
        params = torch.zeros(args.n_high * args.n_wide, 2)
        params[:, 0] = args.c1
        params[:, 1] = args.c2
        force_data = torch.zeros_like(cem_boundary)
        surr_soln, traj_u, traj_f, traj_g = cem.solve(params, cem_boundary,
                      constraint_mask, force_data,
                      step_size=0.1, opt_steps=1000,
                      return_intermediate=True)
        cell_idx = np.random.choice(cem.n_cells)

        self.traj_u = [self.fsm.to_torch(torch.matmul(
            cem.cell_maps.view(-1, cem.cell_maps.size(2)), u_i
        ).view(cem.n_cells, -1, 2)[cell_idx]) for u_i in traj_u[1:]]

        deltas = [
            torch.norm(self.traj_u[i] - self.traj_u[i - 1]).data.cpu().numpy()
            for i in range(1, len(self.traj_u))
        ]

        deltas = [0.0] + deltas
        buckets = np.cumsum(deltas)
        self.buckets = buckets / buckets[-1]

        self.stepsize = 1.0 / args.anneal_steps
        self.steps = 0
        self.base_relax = self.args.relaxation_parameter
        self.guess = solve(self.fem, args,
                           self.traj_u[0], fa.Function(self.fsm.V).vector(),
                           torch.zeros_like(self.traj_u[0]), 50, 0.9)


    def get_weighted_data(self, factor):
        idx = np.searchsorted(self.buckets, factor) - 1
        u1 = self.traj_u[idx]
        u2 = self.traj_u[idx + 1]

        residual = factor - self.buckets[idx]

        alpha = residual / (self.buckets[idx + 1] - self.buckets[idx] + 1e-7)

        u = alpha * u2 + (1.0 - alpha) * u1
        return torch.Tensor(u.data)


@ray.remote(resources={"WorkerFlags": 0.3})
class DeployCollector(DeployCollectorBase):
    pass


if __name__ == '__main__':
    from ..arguments import parser
    import pdb
    args = parser.parse_args()
    pde = Metamaterial(args)
    fsm = FunctionSpaceMap(pde.V, args.bV_dim, args=args)
    net = FeedForwardNet(args, fsm)
    dcollector = DeployCollectorBase(args, 0, net.state_dict())
    collector = CollectorBase(args, 0)
    pdb.set_trace()
