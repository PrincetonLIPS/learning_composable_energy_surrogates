import torch
from .. import fa_combined as fa
from ..pde.metamaterial import make_metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..nets.feed_forward_net import FeedForwardNet
from ..viz.plotting import plot_boundary
from ..energy_model.composed_energy_model import ComposedEnergyModel
from ..energy_model.composed_fenics_energy_model import ComposedFenicsEnergyModel

import math
import matplotlib.pyplot as plt
import io
from PIL import Image

import numpy as np
import pdb


RVES_WIDTH = 4


class EvaluatorBase(object):
    def __init__(self, args, seed):
        self.args = args
        if seed > 0:
            np.random.seed(seed)
            make_p(self.args)
        else:
            self.args.c1 = 0.0
            self.args.c2 = 0.0
        self.pde = make_metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, cuda=False, args=args)
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

        return ((error_V, img_buf, step), (self.args.c1, self.args.c2))

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


class CompressionEvaluatorBase(object):
    def __init__(self, args, seed):
        self.args = args
        if seed != 0:
            np.random.seed(seed)
            make_p(self.args)
        else:
            self.args.c1 = 0.
            self.args.c2 = 0.

        print("Starting compression evaluator with seed {}, c1 {:.3g}, c2 {:.3g}".format(
            seed, self.args.c1, self.args.c2
        ))
        self.pde = make_metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, cuda=False, args=args)
        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)

        self.net = FeedForwardNet(self.args, self.fsm)

        self.sem = SurrogateEnergyModel(self.args, self.net, self.fsm)
        self.cem = ComposedEnergyModel(self.args, self.sem, RVES_WIDTH, RVES_WIDTH)

        c1s = np.ones(RVES_WIDTH*RVES_WIDTH) * self.args.c1
        c2s = np.ones(RVES_WIDTH*RVES_WIDTH) * self.args.c2
        cfem = ComposedFenicsEnergyModel(args, RVES_WIDTH, RVES_WIDTH,
                                         c1s, c2s)

        print("Built energy models for compression evaluator")
        constrained_sides = [True, False, True, False]

        MAX_DISP = args.deploy_disp
        ANNEAL_STEPS = args.anneal_steps
        init_guess = fa.Function(cfem.pde.V).vector()
        for i in range(ANNEAL_STEPS):
            # print("Anneal {} of {}".format(i+1, ANNEAL_STEPS))
            print("Compression evaluator anneal step {}/{}".format(i, ANNEAL_STEPS))
            fenics_boundary_fn = fa.Expression(('0.0', 'X*x[1]'),
                                           element=self.pde.V.ufl_element(),
                                            X=MAX_DISP*(i+1)/ANNEAL_STEPS)

            true_soln = cfem.solve(args=args, boundary_fn=fenics_boundary_fn,
                                   constrained_sides=constrained_sides,
                                   initial_guess=init_guess)
            init_guess = true_soln.vector()

        self.true_soln = true_soln
        self.true_f = cfem.pde.energy(true_soln)
        self.true_scaled_f = cfem.pde.energy(fa.project(fenics_boundary_fn,
                                             cfem.pde.V))

        self.init_boundary_data = torch.zeros(len(self.cem.global_coords), 2)
        self.init_boundary_data[:, 1] = MAX_DISP * torch.Tensor(self.cem.global_coords)[:, 1]

        self.true_soln_points = torch.Tensor([
            self.true_soln(*x)
            for x in self.cem.global_coords
        ])
        self.params = torch.zeros(RVES_WIDTH*RVES_WIDTH, 2)
        self.params[:, 0] = self.args.c1
        self.params[:, 1] = self.args.c2
        self.cem_constraint_mask = torch.zeros(len(self.cem.global_coords))
        self.cem_constraint_mask[self.cem.bot_idxs()] = 1.0
        self.cem_constraint_mask[self.cem.top_idxs()] = 1.0
        self.force_data = torch.zeros(len(self.cem.global_coords), 2)
        print("Finished building CompressionEvaluator")

    def step(self, state_dict, step):
        if state_dict is not None:
            self.cem.sem.net.load_state_dict(state_dict)

        surr_soln, traj_u, traj_f, traj_g = self.cem.solve(self.params, self.init_boundary_data,
                              self.cem_constraint_mask, self.force_data,
                              step_size=0.1, opt_steps=500, return_intermediate=True)

        traj_u_interp = [traj_u[0]]
        traj_f_interp = [traj_f[0]]
        traj_g_interp = [traj_g[0]]
        T = len(traj_u)
        L = 12
        for i in range(1, L):
            t = T * i/L
            idx = int(math.floor(t))
            rem = t - idx
            traj_u_interp.append((1.-rem) * traj_u[idx] + rem * traj_u[idx+1])
            traj_f_interp.append((1.-rem) * traj_f[idx] + rem * traj_f[idx+1])
            traj_g_interp.append((1.-rem) * traj_g[idx] + rem * traj_g[idx+1])

        traj_u_interp.append(traj_u[-1])
        traj_f_interp.append(traj_f[-1])
        traj_g_interp.append(traj_g[-1])

        img_buf = self.visualize_trajectory(
            traj_u_interp, traj_f_interp, traj_g_interp
        )

        assert all(i==j for i, j in zip(self.true_soln_points.size(),
                                        surr_soln.size()))

        err = ((surr_soln-self.true_soln_points)**2).sum().item()

        return ((err, img_buf, step), (self.args.c1, self.args.c2))

    def visualize_trajectory(
        self,
        traj_u,
        traj_f,

        traj_g
    ):
        nrows = int(len(traj_u))
        assert len(traj_u) == len(traj_f)
        assert len(traj_g) == len(traj_u)

        # traj_u = surrogate.fsm.to_ring(torch.cat(traj_u, dim=0))

        fig, axes = plt.subplots(nrows, 1, figsize=(6, 6 * nrows))

        # if nrows > 1:
        #     axes = [ax for axs in axes for ax in axs]
        # else:
        #     axes = [ax for ax in axes]
        # true_solution_fn = surrogate.fsm.get_query_fn(true_solution)
        # proj_true_soln_fn = surrogate.fsm.get_query_fn(surrogate.fsm.to_ring(true_solution))
        initial_coords = np.array(self.cem.global_coords)

        fhat_on_true = self.cem.energy(self.true_soln_points, self.params, None).item()

        fhat_on_scaled = self.cem.energy(self.init_boundary_data, self.params, None).item()


        for i, ax in enumerate(axes):
            if i >= len(traj_u):
                break
            plt.sca(ax)
            ax.scatter(initial_coords[:, 0], initial_coords[:, 1], color='silver',
                        label='rest', alpha=0.2)
            ax.scatter(initial_coords[:, 0] + self.init_boundary_data[:, 0].data.numpy(),
                        initial_coords[:, 1] + self.init_boundary_data[:, 1].data.numpy(),
                        color='k', label='scaled, f={:.2e}, fhat={:.2e}'.format(
                            self.true_scaled_f, fhat_on_scaled
                        ))
            ax.scatter(initial_coords[:, 0] + self.true_soln_points[:, 0].data.numpy(),
                        initial_coords[:, 1] + self.true_soln_points[:, 1].data.numpy(),
                        color='blue', label='true solution, f={:.2e}, fhat={:.2e}'.format(
                            self.true_f, fhat_on_true
                        ))
            ax.scatter(initial_coords[:, 0] + traj_u[i][:, 0].data.numpy(),
                        initial_coords[:, 1] + traj_u[i][:, 1].data.numpy(),
                        color='red', label='surrogate solution, fhat={:.2e}, ||ghat||={:.2e}'.format(
                            traj_f[i], traj_g[i].norm()
                        ))

            fa.plot(self.true_soln, mode='displacement', alpha=0.2)

            ax.legend()
        # plt.show()
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()
