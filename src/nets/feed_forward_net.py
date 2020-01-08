import torch
from torch import nn
from torch.nn import functional as torchF
import ast
from ..geometry.polar import SemiPolarizer
from ..geometry.remove_rigid_body import RigidRemover
import pdb

nonlinearities = {
    "selu": torchF.selu,
    "relu": torchF.relu,
    "elu": torchF.elu,
    "sin": torch.sin,
    "tanh": torchF.tanh,
    "swish1": lambda x, beta: x * torchF.sigmoid(x),
    "swish": lambda x, beta: x * torchF.sigmoid(x * beta),
    "sigmoid": lambda x: torchF.sigmoid(x),
}

inits = {
    "kaiming": torch.nn.init.kaiming_uniform_,
    "orthogonal": torch.nn.init.orthogonal,
    "xavier": torch.nn.init.xavier_uniform,
}


class MovingAverageNormalzier(nn.Module):
    def __init__(self, alpha, dims, fixed=True):
        super(MovingAverageNormalzier, self).__init__()
        self.alpha = alpha
        self.mean = nn.Parameter(torch.zeros(1, *dims), requires_grad=False)
        self.var = nn.Parameter(torch.ones(1, *dims), requires_grad=False)
        self.fixed = fixed

    def forward(self, x):
        # pdb.set_trace()
        out = (x - self.mean) / torch.sqrt(self.var + 1e-12)
        if self.training and not self.fixed:
            batch_mean = torch.mean(x, dim=0, keepdims=True).data
            self.mean.data = (
                1.0 - self.alpha
            ) * batch_mean + self.alpha * self.mean.data
            batch_var = torch.sum((x - self.mean) ** 2, dim=0, keepdims=True).data
            self.var.data = (1.0 - self.alpha) * batch_var + self.alpha * self.var.data

        return out


class FeedForwardNet(nn.Module):
    def __init__(self, args, fsm):
        super(FeedForwardNet, self).__init__()

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
        self.input_dim = fsm.vector_dim + 2
        self.sizes = [self.input_dim] + sizes + [1]
        if args.nonlinearity == 'swish':
            self.betas = nn.ModuleList([
                nn.Parameter(torch.ones(1, self.sizes[i+1]))
                for i in range(len(self.sizes) - 1)
            ])
        self.nonlinearity = nonlinearities[args.nonlinearity]
        self.init = inits[args.init]
        self.dropout = nn.Dropout(p=args.drop_prob)
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.sizes[i], self.sizes[i + 1], bias=bias)
                for i in range(len(self.sizes) - 1)
            ]
        )
        self.output_scale = nn.Parameter(torch.Tensor([[1.0]]), requires_grad=False)
        # for l in self.layers:
        #     self.init(l.weight)

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
        for i, l in enumerate(self.layers):
            x = l(a)
            if self.args.nonlinearity == 'swish':
                a = self.nonlinearity(x, self.betas[i])
            else:
                a = self.nonlinearity(x)
            a = self.dropout(a)
        out = x.view(-1, 1)
        return out
