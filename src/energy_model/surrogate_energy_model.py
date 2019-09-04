"""EnergyModel which computes f with forward pass of a NN"""
# TODO: Change this to NeuralEnergyModel or TorchEnergyModel...

import torch
from torch.autograd import Variable
from copy import deepcopy
import pdb

import numpy as np
import copy

from .. import fa_combined as fa


class SurrogateEnergyModel(object):
    def __init__(self, args, net, function_space_map):
        self.net = net
        self.args = args
        self.fsm = function_space_map

    def prep_inputs(self, inputs):
        return self.fsm.to_torch(inputs)

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
        if force_data is not None:
            energy = energy + self.external_work(boundary_inputs, force_data)
        return energy

    def f_J(self, boundary_inputs, params):
        boundary_inputs = self.prep_inputs(boundary_inputs)
        if params is not None:
            params = self.fsm._cuda(params)
        energy = self.f(boundary_inputs, params)
        jac = torch.autograd.grad(sum(energy),
                                  boundary_inputs,
                                  create_graph=True,
                                  retain_graph=True)[0]
        jac = jac.contiguous()
        return energy, jac

    def f_J_Hvp(self, boundary_inputs, params, vectors):
        raise Exception("Currently deprecated")
        boundary_inputs = self.prep_inputs(boundary_inputs)
        if params is not None:
            params = self.fsm._cuda(params)
        vectors = self.fsm._cuda(vectors)
        energy, jac = self.f_J(boundary_inputs, params)
        jac = jac.contiguous()
        jac = self.fsm.to_torch(jac)
        jvp = jac * vectors

        hvp = torch.autograd.grad(torch.sum(jvp),
                                  boundary_inputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        hvp = hvp.contiguous()
        hvp = self.fsm.to_torch(hvp)

        return energy, jac, hvp

    def f_J_H(self, single_boundary_input, single_param):
        single_boundary_input = self.prep_inputs(single_boundary_input)
        if single_param is not None:
            single_param = self.fsm._cuda(single_param)
        input_replicas = [
            deepcopy(single_boundary_input) for _ in range(self.fsm.vector_dim)
        ]
        boundary_inputs = torch.stack(input_replicas, dim=0)
        param_replicas = [
            deepcopy(single_param) for _ in range(self.fsm.vector_dim)
        ]
        params = torch.stack(param_replicas, dim=0)

        energy, jac = self.f_J(boundary_inputs, params)

        hess = torch.autograd.grad(torch.trace(jac),
                                   boundary_inputs,
                                   create_graph=True,
                                   retain_graph=True)[0]

        hess = hess.contiguous().view(self.fsm.vector_dim, self.fsm.vector_dim)

        return energy[0].view(1, 1), jac[0].contiguous().view(
            1, self.fsm.vector_dim), hess

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

    def solve(self, params, boundary_data, constraint_mask, force_data, *args,
              **kwargs):
        '''
        Inputs:
            params:           [batch_size, n_params] tensor
            boundary_data:    [batch_size, n_locs, geometric_dim] tensor
            constraint_mask:  [batch_size, n_locs] binary mask.
                              True where constrained.
            force_data:       [batch_size, n_locs, geometric_dim] tensor
        '''
        # All inputs should have the same first dimension (batch_size)
        assert len(params) == len(boundary_data)
        assert len(params) == len(constraint_mask)
        assert len(params) == len(force_data)

        # Check boundary, init are all batched ring data
        assert all(
            [self.fsm.is_torch(x) for x in [boundary_data, boundary_data[0]]])

        # Check force
        #assert all([self.fsm.is_force(force_data),
        #            self.fsm.is_force(force_data[0])])

        # Check params
        #assert all([self.fsm.is_param(params), self.fsm.is_param(params[0])])

        # Run either solve_adam or solve_lbfgs
        if self.args is not None and (self.args.solve_optimizer == 'adam'
                                      or self.args.solve_optimizer == 'sgd'):
            return self.solve_adam(params, boundary_data, constraint_mask,
                                   force_data, *args, **kwargs)
        else:
            return self.solve_lbfgs(params, boundary_data, constraint_mask,
                                    force_data, *args, **kwargs)

    def solve_lbfgs(self,
                    params,
                    boundary_data,
                    constraint_mask,
                    force_data,
                    return_intermediate=False,
                    opt_steps=None):
        if opt_steps is None:
            opt_steps = self.args.solve_lbfgs_steps
        x = self.fsm.to_torch(boundary_data).data.detach().clone()
        constraint_mask = self.fsm.to_torch(constraint_mask)
        if self.fsm.cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=True)

        def obj_fn(x_):
            boundary = boundary_data.detach().clone()
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1. - constraint_mask) * x_
            return self.f(f_inputs, params, force_data=force_data)

        optimizer = torch.optim.LBFGS([x], lr=self.args.solve_lbfgs_stepsize,
                                      max_iter=opt_steps)

        traj_u = []
        traj_f = []

        def closure():
            #pdb.set_trace()
            optimizer.zero_grad()
            f = obj_fn(x)
            if return_intermediate:
                traj_u.append(x.detach().clone())
                traj_f.append(f.detach().clone())
            loss = torch.sum(f)
            loss.backward()
            return loss

        optimizer.step(closure)
        #pdb.set_trace()

        if return_intermediate:
            traj_u.append(x.detach().clone())
            traj_f.append(obj_fn(x).detach().clone())

        if return_intermediate:
            return x, traj_u, traj_f
        else:
            return x

    def solve_adam(self,
                   params,
                   boundary_data,
                   constraint_mask,
                   force_data,
                   return_intermediate=False,
                   opt_steps=None):
        if opt_steps is None:
            opt_steps = self.args.solve_steps
        x = self.fsm.to_torch(boundary_data).data.detach().clone()
        constraint_mask = self.fsm.to_torch(constraint_mask)
        if self.fsm.cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=True)

        def obj_fn(x_):
            boundary = boundary_data.detach().clone()
            f_inputs = torch.zeros_like(constraint_mask)
            f_inputs = f_inputs + constraint_mask * boundary
            f_inputs = f_inputs + (1. - constraint_mask) * x_
            # torch_inputs = self.fsm.to_torch(f_inputs)
            return self.f(f_inputs, params, force_data=force_data)

        if self.args.solve_optimizer == 'adam':
            optimizer = torch.optim.Adam([x], lr=self.args.solve_adam_stepsize)
        else:
            optimizer = torch.optim.SGD([x], lr=self.args.solve_sgd_stepsize)

        traj_u = []
        traj_f = []

        for i in range(opt_steps):
            optimizer.zero_grad()
            objective = obj_fn(x)
            if return_intermediate:
                traj_u.append(x.detach().clone())
                traj_f.append(objective.detach().clone())
            torch.sum(objective).backward()
            optimizer.step()

        if return_intermediate:
            if len(traj_u) > 20:
                traj_u = [traj_u[0]] + traj_u[::int(len(traj_u) / 20)]
                traj_f = [traj_f[0]] + traj_f[::int(len(traj_f) / 20)]
            traj_u.append(x.detach().clone())
            traj_f.append(obj_fn(x).detach().clone())
        '''
        for idx in self.constrained_idxs:
            assert np.all(np.isclose(x.data[idx].cpu().numpy(),
                              self.boundary_ring_data.data[idx].cpu().numpy()))
        '''
        if return_intermediate:
            return x, traj_u, traj_f
        else:
            return x
