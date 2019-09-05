import torch
from torch import nn
from torch.nn import functional as torchF
import pdb
import ast

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


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, args, preproc=None):
        super(FeedForwardNet, self).__init__()
        self.args = args
        sizes = ast.literal_eval(args.ffn_layer_sizes)
        bias = args.use_bias
        self.nonlinearity = nonlinearities[args.nonlinearity]
        self.init = inits[args.init]
        self.input_dim = input_dim
        self.sizes = [self.input_dim] + sizes + [1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.sizes[i], self.sizes[i + 1], bias=bias)
                for i in range(len(self.sizes) - 1)
            ]
        )
        self.output_scale = nn.Parameter(torch.Tensor([1.0]))
        self.preproc = preproc
        for l in self.layers:
            # torch.nn.init.orthogonal(l.weight)
            self.init(l.weight)  # , gain=nn.init.calculate_gain('relu'))
            l.weight.data = l.weight.data * args.init_gain

    def initialize_whitening(self, train_x_mean, train_x_std, train_f_mean):
        self.x_mean = nn.Parameter(train_x_mean.view(1, -1), requires_grad=False)
        self.x_std = nn.Parameter(train_x_std.view(1, -1), requires_grad=False)
        self.output_scale = nn.Parameter(train_f_mean.view(1), requires_grad=False)

    def forward(self, boundary_params, params=None):
        if self.preproc is not None:
            boundary_params = self.preproc(boundary_params)
        x = (boundary_params - self.x_mean) / self.x_std
        if params is not None:
            x = torch.cat((x, params), dim=1)
        else:
            x = x
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
        out = (x ** 2).view(-1)
        if self.args.quadratic_scale:
            quadratic_scale = torch.mean(boundary_params ** 2, dim=1).view(-1)
            return out * quadratic_scale * self.output_scale
        else:
            return out * self.output_scale
