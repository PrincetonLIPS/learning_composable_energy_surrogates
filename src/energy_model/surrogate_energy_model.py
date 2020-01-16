"""EnergyModel which computes f with forward pass of a NN"""
# TODO: Change this to NeuralEnergyModel or TorchEnergyModel...

import torch
from torch.autograd import Variable
from copy import deepcopy
import pdb

import numpy as np
import copy

from .. import fa_combined as fa


class EnergyScaler(object):
    def __init__(self, args, preproc, fsm):
        self.quadratic_scale = args.quadratic_scale
        self.log_scale = args.log_scale
        self.fenics_scale = args.fenics_scale
        self.preproc = preproc
        self.fsm = fsm

    def scale(self, f, bparams):
        if self.fenics_scale:
            # pdb.set_trace()
            f = f / torch.Tensor(
                [
                    float(self.fsm.small_pde.energy(self.fsm.to_small_V(bparams[i])))
                    for i in range(len(bparams))
                ]
            ).view(-1, 1)
        elif self.quadratic_scale:
            f = f / torch.sum(
                self.preproc(bparams) ** 2,
                dim=tuple(i for i in range(1, len(bparams.size()))),
                keepdims=True,
            )

        if self.log_scale:
            f = torch.log(1e6 * f) - torch.log(torch.ones_like(f) * 1e6)

        return f

    def descale(self, f, bparams):
        if self.log_scale:
            f = torch.exp(f)
        if self.fenics_scale:
            # pdb.set_trace()
            f = f * torch.Tensor(
                [
                    float(self.fsm.small_pde.energy(self.fsm.to_small_V(bparams[i])))
                    for i in range(len(bparams))
                ]
            ).view(-1, 1)
        elif self.quadratic_scale:
            f = f * torch.sum(
                self.preproc(bparams) ** 2,
                dim=tuple(i for i in range(1, len(bparams.size()))),
                keepdims=True,
            )

        return f


class SurrogateEnergyModel(object):
    def __init__(self, args, net, function_space_map):
        self.net = net
        self.args = args
        self.fsm = function_space_map
        self.scaler = EnergyScaler(args, self.net.preproc, self.fsm)

    def prep_inputs(self, inputs):
        inputs = self.fsm.to_torch(inputs)
        return self.fsm.make_requires_grad(inputs)

    def parameters(self):
        return self.net.parameters()

    def external_work(self, boundary_inputs, force_data):
        return -torch.sum(boundary_inputs * force_data)

    def f(self, boundary_inputs, params, force_data=None):
        if params is not None:
            params = self.fsm._cuda(params)
        boundary_inputs = self.prep_inputs(boundary_inputs)
        if force_data is not None:
            force_data = self.prep_inputs(force_data)
        energy = self.net(boundary_inputs, params)
        energy = self.scaler.descale(energy, boundary_inputs)
        if force_data is not None:
            energy = energy + self.external_work(boundary_inputs, force_data)
        return energy

    def f_J(self, boundary_inputs, params):
        boundary_inputs = self.prep_inputs(boundary_inputs)
        if params is not None:
            params = self.fsm._cuda(params)
        energy = self.f(boundary_inputs, params)
        jac = torch.autograd.grad(
            sum(energy), boundary_inputs, create_graph=True, retain_graph=True
        )[0]
        jac = jac.contiguous()
        return energy, jac

    def f_J_Hvp(self, boundary_inputs, params, vectors):
        boundary_inputs = self.prep_inputs(boundary_inputs)
        energy, jac = self.f_J(boundary_inputs, params)
        vectors = self.fsm.to_torch(vectors)
        jvp = torch.sum(jac * vectors)
        hvp = torch.autograd.grad(
            jvp, boundary_inputs, create_graph=True, retain_graph=True
        )[0]
        hvp = hvp.contiguous()
        return energy, jac, hvp

    def f_J_H(self, single_boundary_input, single_param):
        single_boundary_input = self.prep_inputs(single_boundary_input)
        if single_param is not None:
            single_param = self.fsm._cuda(single_param)
        input_replicas = [
            deepcopy(single_boundary_input) for _ in range(self.fsm.vector_dim)
        ]
        boundary_inputs = torch.stack(input_replicas, dim=0)
        param_replicas = [deepcopy(single_param) for _ in range(self.fsm.vector_dim)]
        params = torch.stack(param_replicas, dim=0)

        energy, jac = self.f_J(boundary_inputs, params)

        hess = torch.autograd.grad(
            torch.trace(jac), boundary_inputs, create_graph=True, retain_graph=True
        )[0]

        hess = hess.contiguous().view(self.fsm.vector_dim, self.fsm.vector_dim)

        return (
            energy[0].view(1, 1),
            jac[0].contiguous().view(1, self.fsm.vector_dim),
            hess,
        )

    def apply_boundary(self, ring_data, constrained_idxs):
        """
        Input:
            ring_data:
                Torch tensor of ring type, is_ring=True
            constrained_idxs:
                array of indices of ring which are constrained
        """
        assert self.fsm.is_ring(ring_data)
        self.boundary_ring_data = ring_data
        self.constrained_idxs = constrained_idxs

    def solve(
        self, params, boundary_data, constraint_mask, force_data, *args, **kwargs
    ):
        """
        Inputs:
            params:           [batch_size, n_params] tensor
            boundary_data:    [batch_size, n_locs, geometric_dim] tensor
            constraint_mask:  [batch_size, n_locs] binary mask.
                              True where constrained.
            force_data:       [batch_size, n_locs, geometric_dim] tensor
        """
        # All inputs should have the same first dimension (batch_size)
        assert len(params) == len(boundary_data)
        assert len(params) == len(constraint_mask)
        assert len(params) == len(force_data)

        # Check boundary, init are all batched ring data
        assert all([self.fsm.is_torch(x) for x in [boundary_data, boundary_data[0]]])

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
        x = self.fsm.to_torch(boundary_data).data.detach().clone()
        constraint_mask = self.fsm.to_torch(constraint_mask).data
        if self.fsm.cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=True)
        boundary = x.detach().clone().data
        force_data = force_data.data.detach().clone()

        def obj_fn(x_):
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1.0 - constraint_mask) * x_
            return self.f(f_inputs, params, force_data=force_data)

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
        x = self.fsm.to_torch(boundary_data).data.detach().clone()
        constraint_mask = self.fsm.to_torch(constraint_mask).data.detach().clone()
        if self.fsm.cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=True)
        boundary = x.data.detach().clone()
        force_data = force_data.data.detach().clone()

        def obj_fn(x_):
            # print("obj_fn")
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1.0 - constraint_mask) * x_
            # torch_inputs = self.fsm.to_torch(f_inputs)
            return self.f(f_inputs, params, force_data=force_data)

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
