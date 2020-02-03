import torch
from torch.autograd import Variable
from copy import deepcopy
import pdb

import numpy as np
import copy

from .. import fa_combined as fa


class ComposedEnergyModel(object):
    def __init__(self, args, surrogate_energy_model, n_high, n_wide):
        self.args = args
        self.sem = surrogate_energy_model
        self.cell_coords = []
        self.n_high = n_high
        self.n_wide = n_wide
        self.n_cells = n_high * n_wide
        for i in range(n_wide):
            for j in range(n_high):
                coords = self.sem.fsm.ring_coords.data.cpu().numpy()
                coords = coords + np.array([[float(i), float(j)]])
                self.cell_coords.append(coords)

        self.global_coords = []
        for coords in self.cell_coords:
            for coord in coords:
                # pdb.set_trace()
                if len(self.global_coords) < 1:
                    self.global_coords.append(coord)
                else:
                    cond = np.isclose(self.global_coords, coord.reshape(1, -1))
                    if not np.any(cond[:, 0] * cond[:, 1]):
                        self.global_coords.append(coord)

        cell_maps = []
        for coords in self.cell_coords:
            cell_map = np.zeros((len(coords), len(self.global_coords)))
            for i, c1 in enumerate(coords):
                for j, c2 in enumerate(self.global_coords):
                    if np.allclose(c1, c2):
                        cell_map[i, j] = 1.0
                        break
            cell_maps.append(cell_map)

        self.cell_maps = torch.Tensor(cell_maps)
        if self.sem.fsm.cuda:
            self.cell_maps = self.cell_maps.cuda()

        self.flip_horiz_map = [None for _ in range(len(self.global_coords))]
        Lh = max([x1 for x1, _ in self.global_coords])
        for i, (x1, x2) in enumerate(self.global_coords):
            target_x1 = Lh - x1
            for j, (y1, y2) in enumerate(self.global_coords):
                if np.isclose(target_x1, y1) and np.isclose(x2, y2):
                    self.flip_horiz_map[i] = j
        assert all([m is not None for m in self.flip_horiz_map])

        self.flip_vert_map = [None for _ in range(len(self.global_coords))]
        Lv = max([x2 for _, x2 in self.global_coords])
        for i, (x1, x2) in enumerate(self.global_coords):
            target_x2 = Lv - x2
            for j, (y1, y2) in enumerate(self.global_coords):
                if np.isclose(x1, y1) and np.isclose(target_x2, y2):
                    self.flip_vert_map[i] = j
        assert all([m is not None for m in self.flip_vert_map])

    def flip_horiz(self, coords, flip_about=None):
        assert (len(coords.size()) == 2 and
                coords.size(0) == len(self.global_coords) and
                coords.size(1) == 2)
        if flip_about is not None:
            coords = coords - flip_about
        out = torch.zeros_like(coords)
        for i in range(len(coords)):
            j = self.flip_horiz_map[i]
            out[i, 0] = -coords[j, 0]
            out[i, 1] = coords[j, 1]
        if flip_about is not None:
            coords = coords + flip_about
        return out

    def flip_vert(self, coords, flip_about=None):
        assert (len(coords.size()) == 2 and
                coords.size(0) == len(self.global_coords) and
                coords.size(1) == 2)
        if flip_about is not None:
            coords = coords - flip_about
        out = torch.zeros_like(coords)
        for i in range(len(coords)):
            j = self.flip_vert_map[i]
            out[i, 0] = coords[j, 0]
            out[i, 1] = -coords[j, 1]
        if flip_about is not None:
            coords = coords + flip_about
        return out

    def top_idxs(self):
        out = []
        for i, c in enumerate(self.global_coords):
            if np.isclose(float(self.n_high), c[1]):
                out.append(i)
        return out

    def bot_idxs(self):
        out = []
        for i, c in enumerate(self.global_coords):
            if np.isclose(0, c[1]):
                out.append(i)
        return out

    def lhs_idxs(self):
        out = []
        for i, c in enumerate(self.global_coords):
            if np.isclose(0, c[0]):
                out.append(i)
        return out

    def rhs_idxs(self):
        out = []
        for i, c in enumerate(self.global_coords):
            if np.isclose(float(self.n_wide), c[0]):
                out.append(i)
        return out

    def interpolate(self, fn):
        data = np.zeros_like(self.global_coords)
        for i, (x1, x2) in enumerate(self.global_coords):
            us = fn(x1, x2)
            data[i, :] = us
        return data

    def energy(self, global_coords, params, force_data):
        cell_coords = torch.matmul(
            self.cell_maps.view(-1, self.cell_maps.size(2)), global_coords
        )
        cell_coords = cell_coords.view(self.n_cells, -1, self.sem.fsm.udim)

        if force_data is not None:
            cell_force_data = torch.matmul(
                self.cell_maps.view(-1, self.cell_maps.size(2)), force_data
            )
            cell_force_data = cell_force_data.view(self.n_cells, -1,
                                                   self.sem.fsm.udim)

        else:
            cell_force_data = torch.zeros_like(cell_coords)

        # pdb.set_trace()

        return torch.sum(self.sem.f(cell_coords, params, cell_force_data))

    def solve(
        self, params, boundary_data, constraint_mask, force_data, *args, **kwargs
    ):
        """
        Inputs:
            params:           [n_cells, n_params] tensor
            boundary_data:    [n_global_locs, geometric_dim] tensor
            constraint_mask:  [n_global_locs] binary mask.
                              True where constrained.
            force_data:       [n_global_locs, geometric_dim] tensor
        """
        assert len(params) == self.n_cells
        assert len(params.size()) == 2
        assert len(boundary_data) == len(self.global_coords)
        assert len(boundary_data.size()) == 2
        assert boundary_data.size(1) == 2
        assert len(constraint_mask) == len(self.global_coords)
        assert len(constraint_mask.size()) == 1

        assert params.device == boundary_data.device
        assert constraint_mask.device == boundary_data.device

        # Check boundary, init are all batched ring data
        # assert all([self.fsm.is_torch(x) for x in [boundary_data, boundary_data[0]]])

        # Check force
        # assert all([self.fsm.is_force(force_data),
        #            self.fsm.is_force(force_data[0])])

        # Check params
        # assert all([self.fsm.is_param(params), self.fsm.is_param(params[0])])

        # Run either solve_adam or solve_lbfgs
        if self.args is not None and (
            self.args.solve_optimizer == "adam" or self.args.solve_optimizer == "sgd"
        ):
            return self.solve_adam(
                params, boundary_data, constraint_mask, force_data, *args, **kwargs
            )
        else:
            return self.solve_lbfgs(
                params, boundary_data, constraint_mask, force_data, *args, **kwargs
            )

    def solve_lbfgs(
        self,
        params,
        boundary_data,
        constraint_mask,
        force_data,
        return_intermediate=False,
        opt_steps=None,
        step_size=None,
    ):
        if opt_steps is None:
            opt_steps = self.args.solve_lbfgs_steps
        if step_size is None:
            step_size = self.args.solve_lbfgs_stepsize
        x = Variable(boundary_data.detach().clone(), requires_grad=True)
        constraint_mask = constraint_mask.data.detach().clone().view(-1, 1)
        boundary = x.detach().clone().data
        force_data = force_data.data.detach().clone()

        def obj_fn(x_):
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1.0 - constraint_mask) * x_
            return self.energy(f_inputs, params, force_data=force_data)

        optimizer = torch.optim.LBFGS([x], lr=step_size, max_iter=opt_steps)

        traj_u = []
        traj_f = []
        traj_g = []

        def closure():
            # print("closure")
            optimizer.zero_grad()
            f = obj_fn(x)
            loss = torch.sum(f)
            loss.backward()
            if return_intermediate:
                traj_u.append(x.data.detach().clone())
                traj_f.append(f.data.detach().clone())
                traj_g.append(torch.norm(x.grad.detach().clone()))
            return loss

        optimizer.step(closure)
        # pdb.set_trace()

        if return_intermediate:
            traj_u.append(x.data.detach().clone())
            traj_f.append(obj_fn(x).data.detach().clone())
            traj_g.append(torch.norm(x.grad.detach().clone()))

        if return_intermediate:
            return x, traj_u, traj_f, traj_g
        else:
            return x

    def solve_adam(
        self,
        params,
        boundary_data,
        constraint_mask,
        force_data,
        return_intermediate=False,
        opt_steps=None,
        step_size=None,
    ):
        if opt_steps is None:
            opt_steps = self.args.solve_sgd_steps
        if step_size is None:
            step_size = self.args.solve_sgd_stepsize
        x = boundary_data.data.detach().clone()
        constraint_mask = constraint_mask.data.detach().clone().view(-1, 1)
        x = Variable(x, requires_grad=True)
        boundary = x.data.detach().clone()
        force_data = force_data.data.detach().clone()

        def obj_fn(x_):
            # print("obj_fn")
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1.0 - constraint_mask) * x_
            # torch_inputs = self.fsm.to_torch(f_inputs)
            return self.energy(f_inputs, params, force_data=force_data)

        if self.args.solve_optimizer == "adam":
            optimizer = torch.optim.Adam([x], lr=step_size)
        else:
            optimizer = torch.optim.SGD([x], lr=step_size)

        traj_u = []
        traj_f = []
        traj_g = []

        for i in range(opt_steps):
            optimizer.zero_grad()
            objective = obj_fn(x)
            torch.sum(objective).backward()
            if return_intermediate:
                traj_u.append(x.data.detach().clone())
                traj_f.append(objective.data.detach().clone())
                traj_g.append(torch.norm(x.grad.detach().clone()))

            optimizer.step()

        if return_intermediate:
            if len(traj_u) > 20:
                traj_u = [traj_u[0]] + traj_u[:: int(len(traj_u) / 20)]
                traj_f = [traj_f[0]] + traj_f[:: int(len(traj_f) / 20)]
                traj_g = [traj_g[0]] + traj_g[:: int(len(traj_g) / 20)]

            traj_u.append(x.detach().clone())
            traj_f.append(obj_fn(x).detach().clone())
            traj_g.append(torch.norm(x.grad.detach().clone()))

        """
        for idx in self.constrained_idxs:
            assert np.all(np.isclose(x.data[idx].cpu().numpy(),
                              self.boundary_ring_data.data[idx].cpu().numpy()))
        """
        if return_intermediate:
            return x, traj_u, traj_f, traj_g
        else:
            return x


if __name__ == "__main__":
    from ..arguments import parser
    from ..pde.metamaterial import Metamaterial
    from ..maps.function_space_map import FunctionSpaceMap
    from .surrogate_energy_model import SurrogateEnergyModel
    from ..data.sample_params import make_bc
    from ..data.random_fourier_fn import make_random_fourier_expression
    import numpy as np
    import pdb

    args = parser.parse_args()
    args.quadratic_scale = False
    args.log_scale = False
    args.relaxation_parameter = 0.9
    pde = Metamaterial(args)
    fsm = FunctionSpaceMap(V=pde.V, bV_dim=args.bV_dim, cuda=False)
    net = lambda x, params: (x ** 2).sum()
    net.preproc = lambda x: x

    np.random.seed(0)

    sem = SurrogateEnergyModel(args, net, fsm)

    cem = ComposedEnergyModel(args, sem, 4, 3)
    boundary_expression = make_random_fourier_expression(
        2, 5000, 1.0, 1.0, fsm.V.ufl_element()
    )

    boundary_data = torch.Tensor(cem.interpolate(boundary_expression))
    # params, boundary_data, constraint_mask, force_data
    params = torch.zeros(4 * 3, 1)
    constraint_mask = torch.Tensor(
        [1.0 if i in cem.bot_idxs() else 0.0 for i in range(len(boundary_data))]
    )
    force_data = torch.zeros_like(boundary_data)

    solution, traj_u, traj_f, traj_g = cem.solve(
        params, boundary_data, constraint_mask, force_data, return_intermediate=True
    )

    pdb.set_trace()
