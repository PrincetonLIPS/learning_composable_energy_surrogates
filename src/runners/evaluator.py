import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..nets.feed_forward_net import FeedForwardNet
from ..viz.plotting import plot_boundary

import math
import matplotlib.pyplot as plt
import io
from PIL import Image

import numpy as np

import ray


@ray.remote(resources={"WorkerFlags": 0.5})
class Evaluator(object):
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        self.p = make_p(self.args)
        self.pde = Metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, cuda=False)
        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)

        self.net = FeedForwardNet(args, self.fsm)

    def step(self, state_dict, step):
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
        surr_soln, traj_u, traj_f, traj_g = surrogate.solve(
            params,
            bc * factor,
            constraint_mask,
            force_data * factor,
            return_intermediate=True,
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
            true_energy = self.fem.f(true_soln)
            initial_guess = true_soln.vector()[:]

        img_buf = self.visualize_trajectory(
            surrogate, true_soln, traj_u, traj_f, traj_g, true_energy, constrained_idxs
        )

        surr_soln_V = self.fsm.to_V(surr_soln)

        diff = surr_soln_V - true_soln

        error_V = fa.assemble(fa.inner(diff, diff) * self.fsm.boundary_ds)

        return error_V, img_buf, step

    def visualize_trajectory(
        self,
        surrogate,
        true_solution,
        traj_u,
        traj_f,
        traj_g,
        true_energy,
        constrained_idxs,
    ):
        nrows = int(math.ceil(float(len(traj_u)) / 2))
        assert len(traj_u) == len(traj_f)
        assert nrows > 0

        traj_u = surrogate.fsm.to_ring(torch.cat(traj_u, dim=0))

        fig, axes = plt.subplots(nrows, 2, figsize=(8, 8 * nrows / 2))

        if nrows > 1:
            axes = [ax for axs in axes for ax in axs]
        else:
            axes = [ax for ax in axes]
        true_solution_fn = surrogate.fsm.get_query_fn(true_solution)
        # proj_true_soln_fn = surrogate.fsm.get_query_fn(surrogate.fsm.to_ring(true_solution))
        for i, ax in enumerate(axes):
            if i >= len(traj_u):
                break
            plot_boundary(lambda x: (0, 0), 1000, label="reference", ax=ax)
            # plot_boundary(proj_true_soln_fn, 1000, label='projected_true_solution', ax=ax)
            plot_boundary(
                true_solution_fn,
                1000,
                label="true_solution, f={:.3e}".format(true_energy),
                linestyle="--",
                ax=ax,
            )
            # pdb.set_trace()
            plot_boundary(
                surrogate.fsm.get_query_fn(traj_u[0]),
                1000,
                label="initial, fhat={:.3e}".format(traj_f[0].item()),
                linestyle="dotted",
                ax=ax,
            )
            plot_boundary(
                surrogate.fsm.get_query_fn(traj_u[i]),
                1000,
                label="iter_{}, fhat={:.3e}, ||g||={:.3e}".format(
                    i, traj_f[i].item(), traj_g[i].item()
                ),
                linestyle="-.",
                ax=ax,
            )
            for s in constrained_idxs:
                x1, x2 = surrogate.fsm.s_to_x(s)
                ax.plot(
                    [traj_u[0, s, 0].item() + x1],
                    [traj_u[0, s, 1].item() + x2],
                    marker="o",
                    markersize=3,
                    color="black",
                )
            ax.legend()
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()
