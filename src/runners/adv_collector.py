import torch
from .. import fa_combined as fa
from ..pde.metamaterial import make_metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..data.example import Example
from ..nets.feed_forward_net import FeedForwardNet
from ..geometry.remove_rigid_body import RigidRemover

from torch.distributions.multivariate_normal import MultivariateNormal

import random
import numpy as np
import ray
import copy
import math
import pdb


class AdversarialCollectorBase(object):
    def __init__(self, args, seed, state_dict):
        self.args = args
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2 ** 32))
        make_p(args)
        # self.last_sample = torch.zeros(self.fsm.vector_dim)

        self.pde = make_metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.net = FeedForwardNet(args, self.fsm)
        self.net.load_state_dict(state_dict)
        self.net.eval()

        self.sem = SurrogateEnergyModel(args, self.net, self.fsm)

        self.n = 0
        self.BASE_ITER = 100
        self.BASE_FACTOR = 0.99
        self.rigid_remover = RigidRemover(self.fsm)
        print("Created adv collector, seed {}".format(seed))

    def damped_error(self, u, u0, p, f, J, H):

        assert len(u.size()) == 2
        f = f.view(1)
        J = J.view(1, -1)
        u0 = u0.view(1, -1)

        # Hack to impose the correct grad
        du = u - u0
        f = f + (du * J).sum(dim=1) + torch.matmul(du, torch.matmul(H, du.t())).diag()

        fhat = self.sem.f(u, params=p.unsqueeze(0).expand((len(u)), len(p))).view(-1)
        # print("f ", f.mean().item())
        # print("fhat ", fhat.mean().item())
        f += 1e-9
        fhat += 1e-9
        # pdb.set_trace()
        return torch.nn.functional.mse_loss(torch.log(f), torch.log(fhat)) - 1e-9 * (
            u ** 2
        ).sum(dim=1)

    def solve(
        self, q, guess, q_last=None, max_iter=None, factor=None, recursion_depth=0,
    ):
        # fa.set_log_level(20)
        if max_iter is None:
            max_iter = self.BASE_ITER
        if factor is None:
            factor = self.BASE_FACTOR
        try:
            # print("recursion {}, iter {}, factor {}".format(recursion_depth, max_iter, factor))
            new_args = copy.deepcopy(self.args)
            new_args = new_args
            new_args.max_newton_iter = max_iter
            new_args.relaxation_parameter = factor
            T = 2 if factor == self.BASE_FACTOR else 10
            Z = sum([2 ** i for i in range(T)])
            new_guess = guess
            for i in range(T):
                new_args.atol = (
                    self.args.atol
                )  # 10**(math.log10(args.atol)*2**i / (2**(T-1)))
                new_args.rtol = 10 ** (math.log10(self.args.rtol) * 2 ** i / Z)
                new_args.max_newton_iter = int(math.ceil(2 ** i * max_iter / Z)) + 10
                # print("solve with rtol {} atol {} iter {} factor {} u_norm {} guess_norm {}".format(
                #    new_args.atol, new_args.rtol, new_args.max_newton_iter, new_args.relaxation_parameter, q.norm().item(),
                #    torch.Tensor(guess).norm().item()))
                f, u = self.fem.f(
                    q, initial_guess=new_guess, return_u=True, args=new_args
                )
                new_guess = u.vector()
            # print("energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}".format(f, sq(q), (f+EPS)/sq(q)))
            return u.vector()
        except Exception as e:
            if q_last is None and recursion_depth == 0:
                return self.solve(
                    q,
                    guess,
                    q_last,
                    max_iter,
                    factor=0.1,
                    recursion_depth=recursion_depth + 1,
                )
            elif q_last is None:
                raise e
            elif recursion_depth >= 8:
                # print("Maximum recursion depth exceeded! giving up.")
                raise e
            else:
                # print("recursing due to error, depth {}:".format(recursion_depth+1))
                # print(e)
                # q_mid = q_last + 0.5*(q-q_last)
                new_factor = 0.1  # max(factor*0.5, 0.05)
                new_max_iter = int(
                    5
                    + max_iter
                    * math.log(1.0 - min(0.9, factor))
                    / math.log(1.0 - new_factor)
                )
                # print("new factor {}, new max iter {}".format(new_factor, new_max_iter))

                # guess = solve(q_mid, guess, q_last, max_iter=new_max_iter,
                #               factor=new_factor, recursion_depth=recursion_depth+1)
                # print("first half of recursion {}".format(recursion_depth+1))
                guess = self.solve(
                    (q + q_last) / 2,
                    guess,
                    q_last,
                    max_iter=new_max_iter,
                    factor=new_factor,
                    recursion_depth=recursion_depth + 1,
                )
                # print("second half of recursion {}".format(recursion_depth+1))
                return self.solve(
                    q,
                    guess,
                    (q + q_last) / 2,
                    max_iter=new_max_iter,
                    factor=new_factor,
                    recursion_depth=recursion_depth + 1,
                )

    def step(self, batch):
        # fa.set_log_level(20)

        u, p, f, J, H, Vsmall_guess = batch

        i = np.random.randint(len(u))
        u = u[i]
        p = p[i]
        f = f[i]
        J = J[i]
        H = H[i]

        if f <= 0.0:
            raise Exception("Received too low energy f")

        self.n += 1
        if self.n > 25:
            raise Exception("Successfully exiting after 25 iters.")

        p = p.view(-1)
        self.args.c1 = p[0].item()
        self.args.c2 = p[1].item()
        self.pde = make_metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, args=self.args)
        self.sem.fsm = self.fsm
        self.net.fsm = self.fsm

        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)

        if (
            True
        ):  # Vsmall_guess is None or np.all(np.isclose(Vsmall_guess[i].numpy(), 0., atol=1e-9, rtol=1e-9)):
            # print("starting solve from scratch")
            guess = fa.Function(self.fsm.V).vector()  # self.fsm.to_V(u).vector()
            last_u = torch.zeros_like(u)
            guess = self.solve(u, guess, last_u)
            last_u = u
        else:
            # print("using previous guess")
            uVsmall_guess = fa.Function(self.fsm.small_V)
            # pdb.set_trace()
            assert len(Vsmall_guess[i].numpy()) == len(uVsmall_guess.vector())
            uVsmall_guess.vector().set_local(Vsmall_guess[i].numpy())
            uVsmall_guess.set_allow_extrapolation(True)
            uV_guess = fa.interpolate(uVsmall_guess, self.fsm.V)
            guess = uV_guess.vector()
            guess = self.solve(u, guess)
            last_u = u

        # print("guess norm {}".format(torch.Tensor(guess).norm().item()))

        u0 = u.clone().detach()

        obj = -self.damped_error(u.unsqueeze(0), u0, p, f, J, H)
        # print("error: {:.5e}".format(-obj.mean().item()))

        newton_damp = np.random.uniform(0.0, self.args.adv_newton_damp)

        steps = (
            self.args.adv_newton_steps
            if self.args.adv_newton
            else self.args.adv_gd_steps
        )

        # Randomize number of steps : combination of cheap small deltas and expensive big deltas
        steps = np.random.randint(1, steps)

        stepsize = (
            self.args.adv_newton_stepsize
            if self.args.adv_newton
            else self.args.adv_gd_stepsize
        )
        stepsize = (
            np.random.random() * stepsize
        )  # Randomize length to allow variations at diff lenghts

        for i in range(steps):
            if self.args.verbose:
                print("Step ", i)
            try:
                if self.args.adv_newton:
                    stack_u = torch.autograd.Variable(
                        torch.stack([u.data for _ in range(len(u))], dim=0),
                        requires_grad=True,
                    )

                    # pdb.set_trace()

                    # objective to minimize is the negative of the error
                    if self.args.verbose:
                        print("Pytorch bit")
                    obj = -self.damped_error(stack_u, u0, p, f, J, H)

                    # print("error: {:.5e}".format(-obj.mean().item()))

                    stack_grad = torch.autograd.grad(
                        obj.sum(), stack_u, create_graph=True, retain_graph=True
                    )[0]
                    grad = stack_grad[0]

                    hess = torch.autograd.grad(torch.trace(stack_grad), stack_u)[0]
                    hess = hess.view(len(u), len(u)) + newton_damp * H

                    grad = grad + newton_damp * J

                    eig = torch.symeig(hess).eigenvalues
                    min_eig = torch.min(eig)
                    # print(min_eig)
                    # print(torch.max(eig))
                    if min_eig < 1e-9:
                        hess = hess + (1e-9 - min_eig) * torch.eye(len(u))
                    hinv = torch.cholesky_inverse(hess)

                    delta_u = torch.matmul(hinv, grad.view(-1, 1)).view(-1)
                else:
                    stack_u = torch.autograd.Variable(
                        u.unsqueeze(0).data, requires_grad=True
                    )
                    obj = -self.damped_error(stack_u, u0, p, f, J, H)
                    # print("error: {:.5e}".format(-obj.mean().item()))
                    grad = torch.autograd.grad(obj.sum(), stack_u)[0][0]
                    delta_u = grad

                # print(delta_u.norm())
                # if delta_u.norm() > 1.0:
                delta_u_scaled = delta_u / delta_u.norm()
                # else:
                #    delta_u_scaled = delta_u

                u = (u - stepsize * delta_u_scaled).detach().clone()

                if self.args.verbose:
                    print("Fenics bit")
                success = False
                tries = 0
                while not success:
                    try:
                        guess = self.solve(u, guess)
                        if self.args.adv_newton:
                            f, JV, H = self.fem.f_J_H(u, initial_guess=guess)
                        else:
                            f, JV = self.fem.f_J(u, initial_guess=guess)
                        f = torch.Tensor([f])
                        J = self.fsm.to_torch(JV)
                        u0 = u.clone().detach()
                        success = True
                    except Exception as e:
                        if self.args.verbose:
                            print("reducing size of u for time ", tries)
                        tries += 1
                        if tries > 10:
                            raise e
                        u = (u + last_u) / 2

                last_u = u
            except Exception as e:
                if i == 0:
                    raise e  #  Didn't even get one step
                else:
                    u = u0  #  Iterate from last successful step

        # print("error: {:.5e}".format(-obj.mean().item()))

        # print("guess norm {}".format(torch.Tensor(guess).norm().item()))

        f, JV, H = self.fem.f_J_H(u0, initial_guess=guess)

        J = self.fsm.to_torch(JV)

        new_uV = fa.Function(self.fsm.V)
        new_uV.vector().set_local(guess)
        new_uV.set_allow_extrapolation(True)

        new_Vsmall_guess = fa.interpolate(new_uV, self.fsm.small_V).vector()

        # pdb.set_trace()

        return Example(u0, p, torch.Tensor([f]), J, H, torch.Tensor(new_Vsmall_guess))


@ray.remote(resources={"WorkerFlags": 0.33})
class AdversarialCollector(AdversarialCollectorBase):
    pass


if __name__ == "__main__":
    from ..arguments import parser
    import pdb
    from ..pde.metamaterial import make_metamaterial

    args = parser.parse_args()
    args.c1 = 0.0
    args.c2 = 0.0
    if args.verbose:
        fa.set_log_level(20)
    pde = make_metamaterial(args)
    fsm = FunctionSpaceMap(pde.V, args.bV_dim)
    fem = FenicsEnergyModel(args, pde, fsm)
    net = FeedForwardNet(args, fsm)
    state_dict = net.state_dict()
    for k, v in state_dict.items():
        v.zero_()
    collector = AdversarialCollectorBase(args, 0, state_dict)
    u = torch.zeros(fsm.vector_dim)
    u[0] += 1e-5
    p = torch.zeros(2)
    f, JV, H = fem.f_J_H(u)
    f = torch.Tensor([f])
    J = fsm.to_torch(JV)
    print(u)
    example = Example(u, p, f, J, H, None)
    for i in range(10):
        print(i)
        example = collector.step(
            (
                example.u.unsqueeze(0),
                example.p.unsqueeze(0),
                example.f.unsqueeze(0),
                example.J.unsqueeze(0),
                example.H.unsqueeze(0),
                None if example.guess is None else example.guess.unsqueeze(0),
            )
        )
        print(example[0])
