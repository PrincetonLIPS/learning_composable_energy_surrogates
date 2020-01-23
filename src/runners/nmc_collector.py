import torch
from .. import fa_combined as fa
from ..pde.metamaterial import Metamaterial
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


class NMCCollectorBase(object):
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2**32))
        make_p(args)
        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.fem = FenicsEnergyModel(args, self.pde, self.fsm)
        self.guess = fa.Function(self.fsm.V).vector()
        self.last_sample = torch.zeros(self.fsm.vector_dim)
        self.n = 0
        self.macro = 0.15*torch.randn(2,2)
        self.fJHu = None
        self.BASE_ITER = 50
        self.BASE_FACTOR = 1.0
        self.rigid_remover = RigidRemover(self.fsm)
        self.stepsize = self.args.nmc_stepsize
        self.temp = self.args.nmc_temp

    def logp_macro(self, q):
        # pdb.set_trace()
        # Negative log probability
        q = self.fsm.to_ring(self.rigid_remover(q))
        if len(q.size()) == 2:
            q = q.unsqueeze(0)
        vert = (q[:, self.fsm.top_idxs()].mean(dim=1) - q[:, self.fsm.bottom_idxs()].mean(dim=1))
        horiz = (q[:, self.fsm.lhs_idxs()].mean(dim=1) - q[:, self.fsm.rhs_idxs()].mean(dim=1))
        # pdb.set_trace()
        # print("vert {}, horiz {}, periodic {}".format(vert.item(), horiz.item(), periodic_part_norm(q).item()))
        x = torch.stack([vert, horiz], dim=1).view(q.size(0), -1)
        mu = self.macro.view(1, -1)
        Sigma_inv = mu**2
        # if q.size(0) == 1:
        #     print("x_macro: {}, residual: {}, scaled_res: {}".format(
        # x.data.cpu().numpy(), (x-mu).data.cpu().numpy(), (Sigma_inv*(x-mu)**2).data.cpu().numpy()))
        # pdb.set_trace()
        return 100*(Sigma_inv*(x-mu)**2).sum()

    def dlogp_macro(self, q):
        q = torch.autograd.Variable(q.data, requires_grad=True)
        return torch.autograd.grad(self.logp_macro(q), q)[0]

    def d2logp_macro(self, q):
        qs = torch.autograd.Variable(
            torch.stack([q.data.clone() for _ in range(len(q))], dim=0),
            requires_grad=True)
        hess = torch.autograd.grad(
            torch.trace(
                torch.autograd.grad(self.logp_macro(qs), qs, create_graph=True
                                    )[0].contiguous()),
                qs)[0]
        hess = hess.contiguous().view(len(q), len(q))
        return hess

    def solve(
        self,
        q,
        guess,
        q_last=None,
        max_iter=None,
        factor=None,
        recursion_depth=0,
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
            T = 10
            Z = sum([2 ** i for i in range(T)])
            for i in range(T):
                new_args.atol = (
                    self.args.atol
                )  # 10**(math.log10(args.atol)*2**i / (2**(T-1)))
                new_args.rtol = 10 ** (math.log10(self.args.rtol) * 2 ** i / Z)
                new_args.max_newton_iter = int(math.ceil(2 ** i * max_iter / Z)) + 1
                #print("solve with rtol {} atol {} iter {} factor {} u_norm {} guess_norm {}".format(
                #    new_args.atol, new_args.rtol, new_args.max_newton_iter, new_args.relaxation_parameter, q.norm().item(),
                #    torch.Tensor(guess).norm().item()))
                f, u = self.fem.f(q, initial_guess=guess, return_u=True, args=new_args)
                guess = u.vector()
            # print("energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}".format(f, sq(q), (f+EPS)/sq(q)))
            return u.vector()
        except Exception as e:
            if q_last is None:
                raise e
            elif recursion_depth >= 50:
                #print("Maximum recursion depth exceeded! giving up.")
                raise e
            else:
                #print("recursing due to error, depth {}:".format(recursion_depth+1))
                #print(e)
                # q_mid = q_last + 0.5*(q-q_last)
                new_factor = 0.1 # max(factor*0.5, 0.1)
                new_max_iter = int(
                    1
                    + max_iter
                    * math.log(1.0 - min(0.9, factor))
                    / math.log(1.0 - new_factor)
                )
                #print("new factor {}, new max iter {}".format(new_factor, new_max_iter))

                # guess = solve(q_mid, guess, q_last, max_iter=new_max_iter,
                #               factor=new_factor, recursion_depth=recursion_depth+1)
                if q_last is None:
                    uV = fa.Function(self.fsm.V)
                    uV.vector().set_local(guess)
                    q_last = self.fsm.to_torch(uV)
                #print("first half of recursion {}".format(recursion_depth+1))
                guess = self.solve(
                    (q+q_last)/2,
                    guess,
                    q_last,
                    max_iter=new_max_iter,
                    factor=new_factor,
                    recursion_depth=recursion_depth + 1,
                )
                #print("second half of recursion {}".format(recursion_depth+1))
                return self.solve(
                    q,
                    guess,
                    (q+q_last)/2,
                    max_iter=new_max_iter,
                    factor=new_factor,
                    recursion_depth=recursion_depth + 1,
                )

    def make_V(self, q, guess, q_last=None):
        guess = self.solve(q, guess, q_last)
        f, u = self.fem.f(q, initial_guess=guess, return_u=True)
        return f + self.logp_macro(q), guess

    def make_d2Vdq(self, q, guess, q_last=None, fJHu=None):
        guess = self.solve(q, guess, q_last=None)
        if fJHu is None:
            f, JV, H, u = self.fem.f_J_H(q, initial_guess=guess, return_u=True)
            J = self.fsm.to_torch(JV)
        else:
            f, J, H, u = fJHu

        dVdq = J + self.dlogp_macro(q)
        d2Vdq = H + self.d2logp_macro(q)

        return d2Vdq, dVdq, u.vector()

    def step(self):
        self.n += 1
        if self.n > 25:
            self.__init__(self.args, np.random.randint(2 ** 32))

        #def nmc(last_sample, last_guess):
        H, J, guess = self.make_d2Vdq(self.last_sample, self.guess, self.fJHu)

        # pi(q) = e^(-V(q))
        # we have dVdq, d2Vdq, want dlogpi(q), where logpi(q) = -V
        H = -H
        J = -J

        evals, evecs = torch.symeig(H, eigenvectors=True)

        evals[evals>-1e-3] = -1e-3


        H_negdef = torch.matmul(torch.matmul(evecs,
                                             torch.diag(evals)),
                                evecs.t())

        negdef_evals = torch.symeig(H_negdef).eigenvalues
        if torch.max(negdef_evals) > -1e-3:
            H_negdef = H_negdef - torch.eye(len(evals)) * (
                torch.max(negdef_evals) + 1e-3)

        Hinv = torch.matmul(torch.matmul(evecs,
                                         torch.diag(1./evals)),
                            evecs.t())

        # print(self.make_V(self.last_sample-0.0001*J, guess))
        # pdb.set_trace()

        mu = self.last_sample - self.stepsize * torch.matmul(Hinv, J.view(-1, 1)).view(-1)

        # pdb.set_trace()

        new_sample = MultivariateNormal(mu, -self.temp*H_negdef).sample().detach().clone()

        if (new_sample - self.last_sample).norm() > 3e-3:
            new_sample = self.last_sample + (
                new_sample - self.last_sample) * 3e-3 / (
                        new_sample-self.last_sample).norm()

        print(new_sample)
        start_V, _ = self.make_V(self.last_sample, guess)
        new_V, new_guess = self.make_V(new_sample, guess, self.last_sample)
        mu_V, _ = self.make_V(mu, guess, self.last_sample)

        start_logp = -start_V
        new_logp = -new_V
        mu_logp = -mu_V

        f, JV, H, u = self.fem.f_J_H(
            new_sample, initial_guess=new_guess, return_u=True)
        J = self.fsm.to_torch(JV)


        print("start logp ", start_logp)
        print("new logp ", new_logp)
        print("mu logp ", mu_logp)
        print("start p ", torch.exp(start_logp))
        print("new p ", torch.exp(new_logp))
        print("mu p ", torch.exp(mu_logp))

        print("logp gap ", new_logp - start_logp)

        if np.log(np.random.random()) < new_logp - start_logp:
            self.last_sample = new_sample
            self.guess = new_guess
            self.fJHu = (f, J, H, u)
            print("sample accepted")
        else:
            print("sample rejected")

        p = torch.Tensor([self.args.c1, self.args.c2])
        f = torch.Tensor([f])

        if f <= 0.:
            raise Exception("Invalid data point!")

        new_uV = fa.Function(self.fsm.V)
        new_uV.set_local(new_guess)
        new_usmall_guess = torch.Tensor(fa.interpolate(new_uV, self.fsm.small_V).vector())
        return Example(new_sample, p, f, J, H, new_usmall_guess)


@ray.remote(resources={"WorkerFlags": 0.33})
class NMCCollector(NMCCollectorBase):
    pass


class Taylor(object):
    def __init__(self, f, J, H, q0):
        assert len(q0.size()) == 1
        J = J.view(1, -1)
        self.f = f
        self.J = J
        self.H = H
        self.q0 = q0

    def __call__(self, qprime):
        q0 = self.q0.view(1, -1)
        if len(qprime.size()) == 1:
            qprime = qprime.view(1, -1)
        bsize = qprime.size(0)
        dq = qprime - q0
        return self.f + (dq*self.J).sum(dim=1) + torch.matmul(
            dq, torch.matmul(self.H, dq.t())).diag() / 2


class NMCOnlineCollectorBase(NMCCollectorBase):
    def __init__(self, args, seed, state_dict):
        NMCCollectorBase.__init__(self, args, seed)
        self.net = FeedForwardNet(args, self.fsm)
        self.net.load_state_dict(state_dict)
        self.net.eval()

        self.sem = SurrogateEnergyModel(args, self.net, self.fsm)

        self.params = torch.Tensor([self.args.c1, self.args.c2])
        self.stepsize = self.args.nmc_online_stepsize
        self.temp = self.args.nmc_online_temp

    def make_V(self, q, guess, q_last=None):
        guess = self.solve(q, guess, q_last)
        f, u = self.fem.f(q, initial_guess=guess, return_u=True)
        # pdb.set_trace()
        f = torch.Tensor([f])
        # f = f.sum().data.cpu().numpy()
        fhat = self.sem.f(q, params=self.params.unsqueeze(0)).view(1)
        f += 1e-9
        fhat += 1e-9
        return - torch.nn.functional.mse_loss(torch.log(f), torch.log(fhat)) + (
            1e-9 * (q**2).sum()
        ), guess

    def make_d2Vdq(self, q, guess, q_last=None, fJHu=None):
        guess = self.solve(q, guess, q_last=None)
        if fJHu is None:
            f, JV, H, u = self.fem.f_J_H(q, initial_guess=guess, return_u=True)
            J = self.fsm.to_torch(JV)
        else:
            f, J, H, u = fJHu

        fhat, Jhat, Hhat = self.sem.f_J_H(q, self.params)
        fhat = fhat.view(1)
        Jhat = Jhat.view(-1)
        Hhat = Hhat.view(len(q), len(q))

        f_taylor = Taylor(f, J, H, q)
        fhat_taylor = Taylor(fhat, Jhat, H, q)
        q = torch.autograd.Variable(q.data, requires_grad=True)

        # pdb.set_trace()

        def energy(q):
            fnew = f_taylor(q) + 1e-9
            fhatnew = fhat_taylor(q) + 1e-9
            # = f/g + g/f
            # d/du = (f'g - fg')/g^2 + (g'f - gf')/f^2
            # d2/du =

            # pdb.set_trace()
            return (torch.log(fnew)-torch.log(fhatnew))**2 + 1e-9 * (q**2).sum(dim=-1)

        # pdb.set_trace()
        qs = torch.stack([torch.autograd.Variable(q.data, requires_grad=True)
                          for _ in range(len(q))])

        energies = energy(qs)
        grads = torch.autograd.grad(energies.sum(), qs,
                                    create_graph=True)[0].contiguous()
        dVdq = -grads[0].data.clone()

        d2Vdq = -torch.autograd.grad(torch.trace(grads), qs)[0].contiguous().view(
            len(q), len(q))

        # Apply damping
        evals = torch.symeig(d2Vdq).eigenvalues
        if torch.min(evals) < 0:
            d2Vdq = d2Vdq + torch.eye(len(dVdq)) * (1e-3 + torch.abs(torch.min(evals)))

        return d2Vdq, dVdq, u.vector()

    def step(self, *args, **kwargs):
        self.n += 1
        if self.n > 10:
            raise Exception("Successfully exiting after 10 iters.")
        return NMCCollectorBase.step(self, *args, **kwargs)


class AdversarialCollectorBase(object):
    def __init__(self, args, seed, state_dict):
        self.args = args
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2**32))
        make_p(args)
        # self.last_sample = torch.zeros(self.fsm.vector_dim)

        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.net = FeedForwardNet(args, self.fsm)
        self.net.load_state_dict(state_dict)
        self.net.eval()

        self.sem = SurrogateEnergyModel(args, self.net, self.fsm)

        self.n = 0
        self.BASE_ITER = 50
        self.BASE_FACTOR = 0.99
        self.rigid_remover = RigidRemover(self.fsm)

    def damped_error(self, u, u0, p, f, J, H):

        assert len(u.size()) == 2
        f = f.view(1)
        J = J.view(1, -1)
        u0 = u0.view(1, -1)

        # Hack to impose the correct grad
        du = u - u0
        f = f + (du*J).sum(dim=1) + torch.matmul(
            du, torch.matmul(H, du.t())).diag()


        fhat = self.sem.f(u, params=p.unsqueeze(0).expand((len(u)), len(p))).view(-1)
        # print("f ", f.mean().item())
        # print("fhat ", fhat.mean().item())
        f += 1e-9
        fhat += 1e-9
        # pdb.set_trace()
        return torch.nn.functional.mse_loss(torch.log(f), torch.log(fhat)) - 1e-9 * (u**2).sum(dim=1)


    def solve(
        self,
        q,
        guess,
        q_last=None,
        max_iter=None,
        factor=None,
        recursion_depth=0,
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
                new_args.max_newton_iter = int(math.ceil(2 ** i * max_iter / Z)) + 1
                # print("solve with rtol {} atol {} iter {} factor {} u_norm {} guess_norm {}".format(
                #    new_args.atol, new_args.rtol, new_args.max_newton_iter, new_args.relaxation_parameter, q.norm().item(),
                #    torch.Tensor(guess).norm().item()))
                f, u = self.fem.f(q, initial_guess=new_guess, return_u=True, args=new_args)
                new_guess = u.vector()
            # print("energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}".format(f, sq(q), (f+EPS)/sq(q)))
            return u.vector()
        except Exception as e:
            if q_last is None and recursion_depth == 0:
                return self.solve(q, guess, q_last, max_iter, factor=0.01)
            elif q_last is None:
                raise e
            elif recursion_depth >= 8:
                # print("Maximum recursion depth exceeded! giving up.")
                raise e
            else:
                # print("recursing due to error, depth {}:".format(recursion_depth+1))
                # print(e)
                # q_mid = q_last + 0.5*(q-q_last)
                new_factor = 0.1 # max(factor*0.5, 0.05)
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
                    (q+q_last)/2,
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
                    (q+q_last)/2,
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

        if f <= 0.:
            raise Exception("Received too low energy f")
        
        self.n += 1
        if self.n > 25:
            raise Exception("Successfully exiting after 25 iters.")

        p = p.view(-1)
        self.args.c1 = p[0].item()
        self.args.c2 = p[1].item()
        self.pde = Metamaterial(self.args)
        self.fsm = FunctionSpaceMap(self.pde.V, self.args.bV_dim, args=self.args)
        self.sem.fsm = self.fsm
        self.net.fsm = self.fsm

        self.fem = FenicsEnergyModel(self.args, self.pde, self.fsm)
        
        if True:#Vsmall_guess is None or np.all(np.isclose(Vsmall_guess[i].numpy(), 0., atol=1e-9, rtol=1e-9)):
            # print("starting solve from scratch")
            guess = fa.Function(self.fsm.V).vector() # self.fsm.to_V(u).vector()
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

        obj = - self.damped_error(u.unsqueeze(0), u0, p, f, J, H)
        # print("error: {:.5e}".format(-obj.mean().item()))

        for i in range(1 if self.args.adv_newton else self.args.adv_gd_steps):
            if self.args.adv_newton:
                stack_u = torch.autograd.Variable(
                    torch.stack([u.data for _ in range(len(u))], dim=0),
                                            requires_grad=True)

                # pdb.set_trace()

                # objective to minimize is the negative of the error
                obj = - self.damped_error(stack_u, u0, p, f, J, H)

                # print("error: {:.5e}".format(-obj.mean().item()))

                stack_grad = torch.autograd.grad(obj.sum(), stack_u,
                                           create_graph=True,
                                           retain_graph=True)[0]
                grad = stack_grad[0]

                hess = torch.autograd.grad(torch.trace(stack_grad), stack_u)[0]
                hess = hess.view(len(u), len(u))

                eig = torch.symeig(hess).eigenvalues
                min_eig = torch.min(eig)
                # print(min_eig)
                # print(torch.max(eig))
                if min_eig < 1e-9:
                    hess = hess + (1e-9 - min_eig) * torch.eye(len(u))
                hinv = torch.cholesky_inverse(hess)
                delta_u = torch.matmul(hinv, grad.view(-1, 1)).view(-1)
            else:
                stack_u = torch.autograd.Variable(u.unsqueeze(0).data,
                                                  requires_grad=True)
                obj = - self.damped_error(stack_u, u0, p, f, J, H)
                # print("error: {:.5e}".format(-obj.mean().item()))
                grad = torch.autograd.grad(obj.sum(), stack_u)[0][0]
                delta_u = grad

            # print(delta_u.norm())
            if delta_u.norm() > 1.0:
                delta_u_scaled = delta_u / delta_u.norm()
            else:
                delta_u_scaled = delta_u

            u = (u - self.args.adv_collector_stepsize * delta_u_scaled).detach().clone()
            
            success = False
            tries = 0
            while not success:
                try:
                    guess = self.solve(u, guess, last_u)
                    if self.args.adv_newton:
                        f, JV, H = self.fem.f_J_H(u, initial_guess=guess)
                    else:
                        f, JV = self.fem.f_J(u, initial_guess=guess)
                    f = torch.Tensor([f])
                    J = self.fsm.to_torch(JV)
                    u0 = u.clone().detach()
                    success = True
                except Exception as e:
                    print("reducing size of u for time ", tries)
                    tries += 1
                    if tries > 10:
                        raise e
                    u = (u + last_u) / 2
        

        # print("error: {:.5e}".format(-obj.mean().item()))

        # print("guess norm {}".format(torch.Tensor(guess).norm().item()))
        
        f, JV, H = self.fem.f_J_H(u, initial_guess=guess)
        
        J = self.fsm.to_torch(JV)

        new_uV = fa.Function(self.fsm.V)
        new_uV.vector().set_local(guess)
        new_uV.set_allow_extrapolation(True)

        new_Vsmall_guess = fa.interpolate(new_uV, self.fsm.small_V).vector()

        # pdb.set_trace()

        return Example(u, p, torch.Tensor([f]), J, H,
                       torch.Tensor(new_Vsmall_guess))


@ray.remote(resources={"WorkerFlags": 0.33})
class NMCOnlineCollector(NMCOnlineCollectorBase):
    pass


@ray.remote(resources={"WorkerFlags": 0.33})
class AdversarialCollector(AdversarialCollectorBase):
    pass


if __name__ == '__main__':
    from ..arguments import parser
    import pdb
    from ..pde.metamaterial import Metamaterial
    args = parser.parse_args()
    args.c1 = 0.
    args.c2 = 0.
    fa.set_log_level(20)
    pde = Metamaterial(args)
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
        example = collector.step((example.u.unsqueeze(0),
                                  example.p.unsqueeze(0),
                                  example.f.unsqueeze(0),
                                  example.J.unsqueeze(0),
                                  example.H.unsqueeze(0),
                                  None if example.guess is None
                                  else example.guess.unsqueeze(0)))
        print(example[0])

    '''
    from ..arguments import parser
    import pdb
    args = parser.parse_args()
    fa.set_log_level(20)
    collector = NMCCollectorBase(args, 0)
    net = FeedForwardNet(args, collector.fsm)
    state_dict = net.state_dict()
    for k, v in state_dict.items():
        v.zero_()

    # print(state_dict)

    online_collector = NMCOnlineCollectorBase(args, 0, state_dict)

    for i in range(10):
        print(online_collector.step()[0])
        # pdb.set_trace()
    '''
