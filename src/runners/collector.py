import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial, PoissonMetamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..data.example import Example
from ..nets.feed_forward_net import FeedForwardNet

import random
import numpy as np
import ray


class CollectorBase(object):
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        make_p(args)
        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.fem = FenicsEnergyModel(args, self.pde, self.fsm)
        self.bc, _, _, self.constraint_mask = make_bc(args, self.fsm)
        self.stepsize = 1.0 / args.anneal_steps
        self.guess = fa.Function(self.fsm.V).vector()
        self.steps = 0
        self.base_relax = self.args.relaxation_parameter

    def increment_factor(self):
        self.steps += 1
        if self.steps > self.args.anneal_steps:
            raise Exception("Collector finished annealing successfully.")
        if self.args.verbose:
            print("Anneal step {}/{}".format(self.steps, self.args.anneal_steps))

    def get_weighted_data(self, factor):
        return self.bc * factor

    def step(self):
        self.increment_factor()
        # if self.steps > self.args.anneal_steps:
        #     raise Exception("Self-destructing; have completed annealing")
        factor = (self.steps + np.random.random() - 0.5) / self.args.anneal_steps
        weighted_data = self.get_weighted_data(factor)
        if self.args.verbose:
            print("Factor: {}".format(factor))
        input_boundary_fn = self.fem.fsm.to_V(weighted_data)

        tries = 0
        success = False
        while not success:
            self.fem.args.relaxation_parameter = self.base_relax / (4 ** tries)
            try:
                if self.args.verbose:
                    print("Try {}/{}".format(tries + 1, 3))
                if self.args.hess:
                    f, JV, H, solution = self.fem.f_J_H(
                        input_boundary_fn, initial_guess=self.guess, return_u=True
                    )
                else:
                    H = torch.zeros(self.fem.fsm.vector_dim, self.fem.fsm.vector_dim)
                    f, JV, solution = self.fem.f_J(
                        input_boundary_fn, initial_guess=self.guess, return_u=True
                    )
                success = True
            except Exception as e:
                print(e)
                tries += 1
                if tries >= 1:
                    raise (e)
        self.fem.args.relaxation_parameter = self.base_relax

        self.guess = solution.vector()
        u = torch.Tensor(weighted_data.data)
        p = torch.Tensor([self.args.c1, self.args.c2])
        f = torch.Tensor([f])
        J = self.fsm.to_torch(JV)

        solution.set_allow_extrapolation(True)
        new_usmall_guess = torch.Tensor(
            fa.interpolate(solution, self.fsm.small_V).vector()
        )

        return Example(u, p, f, J, H, new_usmall_guess)


@ray.remote(resources={"WorkerFlags": 0.33})
class Collector(CollectorBase):
    pass


class PolicyCollectorBase(CollectorBase):
    def __init__(self, args, seed, state_dict):
        CollectorBase.__init__(self, args, seed)

        force_data = make_force(args, self.fsm)

        net = FeedForwardNet(args, self.fsm)
        net.load_state_dict(state_dict)
        net.eval()

        surrogate = SurrogateEnergyModel(args, net, self.fsm)

        params = torch.Tensor([[self.args.c1, self.args.c2]])

        _, self.traj_u, _ = surrogate.solve(
            params, self.bc, self.constraint_mask, force_data, return_intermediate=True
        )

        if args.weight_space_trajectory:
            deltas = [
                torch.norm(self.traj_u[i] - self.traj_u[i - 1]).data.cpu().numpy()
                for i in range(1, len(self.traj_u))
            ]

        else:
            deltas = np.array([1.0 / len(self.traj_u) for _ in range(len(self.traj_u))])

        deltas = [0.0] + deltas
        buckets = np.cumsum(deltas)
        self.buckets = buckets / buckets[-1]

    def get_weighted_data(self, factor):
        idx = np.searchsorted(self.buckets, factor) - 1
        u1 = self.traj_u[idx]
        u2 = self.traj_u[idx + 1]

        residual = factor - self.buckets[idx]

        alpha = residual / (self.buckets[idx + 1] - self.buckets[idx] + 1e-7)

        u = alpha * u2 + (1.0 - alpha) * u1

        return torch.Tensor(u.data)


@ray.remote(resources={"WorkerFlags": 0.5})
class PolicyCollector(PolicyCollectorBase):
    pass


if __name__ == "__main__":
    from ..arguments import parser
    import pdb

    fa.set_log_level(20)
    args = parser.parse_args()
    collector = CollectorBase(args, 0)
    example = collector.step()
    pdb.set_trace()
