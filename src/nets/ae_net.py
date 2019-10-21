import torch
from torch import nn
from torch.nn import functional as torchF
import ast
from ..geometry.polar import SemiPolarizer
from ..geometry.remove_rigid_body import RigidRemover
from ..util.moving_avg_normalizer import MovingAverageNormalzier
import numpy as np



nonlinearities = {
    "selu": torchF.selu,
    "relu": torchF.relu,
    "elu": torchF.elu,
    "sin": torch.sin,
    "tanh": torchF.tanh,
    "swish": lambda x: x * torchF.sigmoid(x) / 1.1,
    "sigmoid": lambda x: torchF.sigmoid(x),
}

inits = {
    "kaiming": torch.nn.init.kaiming_uniform_,
    "orthogonal": torch.nn.init.orthogonal,
    "xavier": torch.nn.init.xavier_uniform,
}


class AutoencoderNet(nn.Module):
    def __init__(self, args, fsm):
        super(AutoencoderNet, self).__init__()

        self.args = args

        self.preproc_fns = []
        if args.remove_rigid:
            self.preproc_fns.append(RigidRemover(fsm))
        if args.semipolarize:
            self.preproc_fns.append(SemiPolarizer(fsm))

        sizes = ast.literal_eval(args.ffn_layer_sizes)
        bias = args.use_bias
        self.normalizer = MovingAverageNormalzier(
            args.normalizer_alpha, (fsm.vector_dim,), self.args.fix_normalizer
        )
        self.nonlinearity = nonlinearities[args.nonlinearity]
        self.init = inits[args.init]
        self.input_dim = fsm.vector_dim + 2
        self.sizes = [self.input_dim] + sizes + sizes[::-1] + [1+fsm.vector_dim*2]
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.sizes[i], self.sizes[i + 1], bias=bias)
                for i in range(len(self.sizes) - 1)
            ]
        )
        self.output_scale = nn.Parameter(torch.Tensor([[1.0]]),
                                         requires_grad=False)
        for l in self.layers:
            self.init(l.weight)

    def preproc(self, x):
        for fn in self.preproc_fns:
            x = fn(x)
        return x

    def forward(self, boundary_params, params=None):
        if self.preproc is not None:
            boundary_params = self.preproc(boundary_params)
        boundary_params = self.normalizer(boundary_params)
        if params is not None:
            x = torch.cat((boundary_params, params), dim=1)
        else:
            x = boundary_params
        if next(self.parameters()).is_cuda:
            if not x.is_cuda:
                x = x.cuda()
        assert self.input_dim == x.shape[1]
        a = x
        for l in self.layers:
            x = l(a)
            if a.size() == x.size():  # Use residual connection if sizes allow
                a = a + self.nonlinearity(x)
            else:
                a = self.nonlinearity(x)
        A = torchF.softplus(x[:, 1+(x.size(1)-1)//2:])
        x = x[:, 1:1+(x.size(1)-1)//2]
        offset = x[:, 0].view(-1, 1)**2
        return offset + torch.sum((boundary_params*A - x*A)**2, dim=1, keepdims=True) * self.output_scale
