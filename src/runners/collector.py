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
class Collector(object):
    def __init__(self, args):
        self.args = args
        make_p(args)
        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.data_V_dim,
                                    args.metamaterial_bV_dim)
        self.fem = FenicsEnergyModel(args, self.pde, self.fsm)
        self.bc, _, _, self.constraint_mask = make_bc(args, self.fsm)
        self.stepsize = 1. / args.anneal_steps
        self.factor = 0.
        self.guess = fa.Function(self.fsm.V).vector()

    def increment_factor(self):
        self.factor = np.min(
            [1., self.factor + self.stepsize * (1 + random.random() - 0.5)])

    def get_weighted_data(self, factor):
        return self.bc * factor

    def step(self):
        self.increment_factor()
        weighted_data = self.get_weighted_data(self.factor)
        input_boundary_fn = self.fem.fsm.to_V(weighted_data)
        f, JV, solution = self.fem.f_J(input_boundary_fn,
                                       initial_guess=self.guess,
                                       return_u=True)

        self.guess = solution.vector()
        u = torch.Tensor(weighted_data.data)
        p = torch.Tensor([self.args.c1, self.args.c2])
        f = torch.Tensor([f])
        J = self.fem.to_torch(JV)

        return Example(u, p, f, J)


@ray.remote
class PolicyCollector(Collector):
    def __init__(self, args, surrogate):
        super(PolicyCollector, self).__init__(args)

        force_data = make_force(args, self.fsm)

        params = torch.Tensor([args.c1, args.c2])

        _, self.traj_u, _ = surrogate.solve(params,
                                            self.bc,
                                            self.constraint_mask,
                                            force_data,
                                            return_intermediate=True)

        if args.weight_space_trajectory:
            deltas = [
                torch.norm(self.traj_u[i] -
                           self.traj_u[i - 1]).data.cpu().numpy()
                for i in range(1, len(self.traj_u))
            ]
            deltas = [0.] + deltas

        else:
            deltas = np.arange(0., 1., 1. / len(self.traj_u))
        buckets = np.cumsum(deltas)
        self.buckets = buckets / buckets[-1]

    def get_weighted_data(self, factor):
        idx = np.searchsorted(self.buckets, factor) - 1
        u1 = self.traj_u[idx]
        u2 = self.traj_u[idx + 1]

        residual = factor - self.buckets[idx]

        alpha = residual / (self.buckets[idx + 1] - self.buckets[idx] + 1e-7)

        u = alpha * u2 + (1. - alpha) * u1

        return torch.Tensor(u.data)
