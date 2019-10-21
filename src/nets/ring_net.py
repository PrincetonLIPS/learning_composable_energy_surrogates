import torch
from torch import nn
from torch.nn import functional as torchF
from torch.autograd import Variable
import pdb
import numpy as np
import ast
from ..geometry.polar import SemiPolarizer
from ..geometry.remove_rigid_body import RigidRemover
from ..util.moving_avg_normalizer import MovingAverageNormalzier

nonlinearities = {
    'selu': torchF.selu,
    'relu': torchF.relu,
    'elu': torchF.elu,
    'swish': lambda x: x * torchF.sigmoid(x) / 1.1,
    'sigmoid': lambda x: torchF.sigmoid(x)
}

inits = {
    'kaiming': torch.nn.init.kaiming_uniform_,
    'orthogonal': torch.nn.init.orthogonal,
    'xavier': torch.nn.init.xavier_uniform
}


class RingNet(nn.Module):
    """Network that uses ring convolutions

    A ring convolution can be implemented as 1d convolution, preceded by
    circular padding."""
    def __init__(self, args, fsm):
        super(RingNet, self).__init__()
        self.args = args
        self.fsm = fsm

        self.preproc_fns = []
        if args.remove_rigid:
            self.preproc_fns.append(RigidRemover(fsm))
        if args.semipolarize:
            self.preproc_fns.append(SemiPolarizer(fsm))
        self.normalizer = MovingAverageNormalzier(
            args.normalizer_alpha, (fsm.vector_dim,), self.args.fix_normalizer
        )
        self.input_dim = 4
        sizes = ast.literal_eval(args.ringnet_layer_sizes)
        bias = True
        self.init = inits[args.init]
        self.nonlinearity = nonlinearities[args.nonlinearity]
        self.sizes = [self.input_dim + 1] + sizes
        self.strides = ast.literal_eval(args.ringnet_strides)
        self.widths = ast.literal_eval(args.ringnet_widths)
        assert len(sizes) == len(self.strides)
        assert len(self.strides) == len(self.widths)
        assert np.prod(self.strides) <= 4 * fsm.elems_along_edge
        assert 4 * fsm.elems_along_edge % np.prod(self.strides) == 0
        n_remain = 4 * fsm.elems_along_edge // np.prod(
            self.strides) * self.sizes[-1]
        self.ffn_sizes = [n_remain] + ast.literal_eval(
            args.ringnet_ffn_layer_sizes) + [1]
        self.layers = nn.ModuleList([
            nn.Conv1d(self.sizes[i],
                      self.sizes[i + 1],
                      self.widths[i],
                      bias=bias,
                      padding=(self.widths[i] - 1),
                      padding_mode='circular',
                      stride=self.strides[i])
            for i in range(len(self.sizes) - 1)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Linear(self.ffn_sizes[i], self.ffn_sizes[i + 1])
            for i in range(len(self.ffn_sizes) - 1)
        ])

        ss = [i for i in range(4 * self.fsm.elems_along_edge)]

        xs = [s_to_x(s, fsm.elems_along_edge) for s in ss]
        # pdb.set_trace()
        rs = np.array([
            np.linalg.norm(np.array([x1, x2]) - np.array([0.5, 0.5]))
            for x1, x2 in xs
        ])
        rs = (rs - rs.mean()) / rs.std()
        self.rs = nn.Parameter(torch.Tensor(rs).view(1, -1, 1),
                               requires_grad=False)
        self.output_scale = nn.Parameter(torch.Tensor([[1.0]]), requires_grad=False)
        for l in self.layers:
            self.init(l.weight)

    def preproc(self, x):
        for fn in self.preproc_fns:
            x = fn(x)
        return x

    def forward(self, boundary_params, params=None):

        boundary_params = self.fsm.to_torch(boundary_params, keep_grad=True)
        if self.preproc is not None:
            boundary_params = self.preproc(boundary_params)
        boundary_params = self.normalizer(boundary_params)

        boundary_params = self.fsm.to_ring(boundary_params, keep_grad=True)
        assert (boundary_params.size(1) % 4 == 0)

        if params is not None:
            params = params.view(boundary_params.size(0), 1, -1)
            params = params.expand(boundary_params.size())
            x = torch.cat((boundary_params, params), dim=2)
        else:
            x = boundary_params
        # pdb.set_trace()
        if next(self.parameters()).is_cuda:
            if not x.is_cuda:
                x = x.cuda()
        rs = self.rs.expand(x.size(0), x.size(1), 1)
        x = torch.cat((x, rs), dim=2)
        x = x.permute(0, 2, 1)
        # pdb.set_trace()
        # pdb.set_trace()
        for l in self.layers:
            a = l(x)
            if a.size() == x.size():
                x = x + self.nonlinearity(a)
            else:
                x = self.nonlinearity(a)
        x = x.view(x.size(0), -1)
        for l in self.ffn_layers:
            a = l(x)
            x = self.nonlinearity(a)
        out = torch.mean(a, dim=1, keepdim=True)**2
        if self.args.quadratic_scale:
            quadratic_scale = torch.mean(torch.mean(boundary_params**2, dim=2),
                                         dim=1).view(-1, 1)
            # pdb.set_trace()
            return out.view(-1, 1) * quadratic_scale * self.output_scale
        else:
            return out.view(-1, 1) * self.output_scale


def s_to_x(s, elems_along_edge):
    '''Apparently this isn't used.'''
    s = s % (elems_along_edge * 4)
    # Bottom
    if s <= elems_along_edge:
        return s / elems_along_edge, 0.
    # RHS
    elif s <= 2 * elems_along_edge:
        return 1.0, (s - elems_along_edge) / elems_along_edge
    # Top
    elif s <= 3 * elems_along_edge:
        return (3 * elems_along_edge - s) / elems_along_edge, 1.0
    else:
        return 0.0, (4 * elems_along_edge - s) / elems_along_edge
