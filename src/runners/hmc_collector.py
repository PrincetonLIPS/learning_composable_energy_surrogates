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

import random
import numpy as np
import ray
import copy
import math


class HMCCollectorBase(object):
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        make_p(args)
        self.pde = Metamaterial(args)
        self.fsm = FunctionSpaceMap(self.pde.V, args.bV_dim, args=args)
        self.fem = FenicsEnergyModel(args, self.pde, self.fsm)
        self.guess = fa.Function(self.fsm.V).vector()
        self.last_sample = torch.zeros(self.fsm.vector_dim)
        self.n = 0

    def step(self):
        self.n += 1
        if self.n > 25:
            self.__init__(self.args, np.random.randint(2**31))
        path_len = np.random.uniform(0.05, 0.2)
        std = np.random.uniform(0.05, 0.2)
        temp = np.random.choice([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        sq_vn = [random.random() for _ in range(2)]
        sq_hn = [random.random() for _ in range(2)]
        sq_vp = [random.random() for _ in range(2)]
        sq_hp = [random.random() for _ in range(2)]
        sq_all = random.random()
        sq_alpha = random.random()
        EPS = 1e-7

        BASE_ITER = 30
        BASE_FACTOR = 1.0
        step_size = np.random.uniform(0.003, 0.02)
        fem = self.fem
        fsm = self.fsm
        args = self.args
        rigid_remover = RigidRemover(fsm)

        def sq(q):
            q = fsm.to_ring(rigid_remover(q))
            vert = (q[fsm.top_idxs()].sum(dim=0) - q[fsm.bottom_idxs()].sum(dim=0))
            vert[0] *= (sq_vn[0] if vert[0] < 0 else sq_vp[0])
            vert[1] *= (sq_vn[1] if vert[1] < 0 else sq_vp[1])
            vert = (vert**2).sum()
            horiz = (q[fsm.lhs_idxs()].sum(dim=0) - q[fsm.rhs_idxs()].sum(dim=0))
            horiz[0] *= (sq_hn[0] if horiz[0] < 0 else sq_hp[0])
            horiz[1] *= (sq_hn[1] if horiz[1] < 0 else sq_hp[1])
            horiz = (horiz**2).sum()
            # pdb.set_trace()
            ret = EPS + sq_all * q.norm()**2 + vert + horiz 
            return ret ** sq_alpha  # + q.norm() # torch.nn.functional.softplus(vert + horiz) # + (q**2).sum()


        def dsq(q):
            q = torch.autograd.Variable(q.data, requires_grad=True)
            return torch.autograd.grad(sq(q), q)[0]

        def solve(q, guess, q_last=None, max_iter=BASE_ITER, factor=BASE_FACTOR, recursion_depth=0):
            try:
                # print("recursion {}, iter {}, factor {}".format(recursion_depth, max_iter, factor))
                new_args = copy.deepcopy(args)
                new_args = new_args
                new_args.max_newton_iter = max_iter
                new_args.relaxation_parameter = factor
                T = 8
                Z = sum([2**i for i in range(T)])
                for i in range(T):
                    new_args.atol = args.atol #10**(math.log10(args.atol)*2**i / (2**(T-1)))
                    new_args.rtol = 10**(math.log10(args.rtol)*2**i / Z)
                    new_args.max_newton_iter = int(math.ceil(2**i * max_iter / Z)) + 1
                    f, u = fem.f(q, initial_guess=guess, return_u=True, args=new_args)
                    guess = u.vector()
                # print("energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}".format(f, sq(q), (f+EPS)/sq(q)))
                return u.vector()
            except Exception as e:
                if q_last is None:
                    raise e
                elif recursion_depth >= 2:
                    print("Maximum recursion depth exceeded! giving up.")
                    raise e
                else:
                    print("recursing due to error:")
                    print(e)
                    # q_mid = q_last + 0.5*(q-q_last)
                    new_factor = factor * 0.3
                    new_max_iter = int(1 + max_iter * math.log(1.-min(0.9, factor))/math.log(1.-new_factor))
                    # guess = solve(q_mid, guess, q_last, max_iter=new_max_iter,
                    #               factor=new_factor, recursion_depth=recursion_depth+1)
                    return solve(q, guess, q_last, max_iter=new_max_iter,
                                 factor=new_factor, recursion_depth=recursion_depth+1)
        
        def make_dVdq(q, guess, q_last=None):
            guess = solve(q, guess, q_last)
            f, JV, u = fem.f_J(q, initial_guess=guess,
                              return_u=True)
            J = fsm.to_torch(JV)
            # (f'g - g'f)/g^2
            dVdq = (J * sq(q) - (f+EPS) * 2 * dsq(q)) / sq(q)**2
            return dVdq, u.vector()
        
        def leapfrog(q, p, guess):
            q = q.clone()
            p = p.clone()
            dVdq, guess = make_dVdq(q, guess, q_last=None)
            p = p - step_size  * dVdq / (2 * temp)  # half step
            failed = False
            for i in range(int(path_len / step_size) - 1):
                print("leapfrog step {}/{}".format(i, int(path_len / step_size) - 1))
                q_last = q
                try:
                    q = q + step_size * p  # whole step
                    dVdq, guess = make_dVdq(q, guess, q_last=q_last)
                    p = p - step_size * (dVdq / temp)  # whole step
                except Exception as e:
                    print("passing leapfrog due to: {}".format(e))
                    q = q_last
                    failed = True
                    break
            q_last = q
            try:
                q = q + step_size * p  # whole step
                if failed:
                    raise Exception("Already failed in for loop")
                dVdq, guess = make_dVdq(q, guess, q_last=q_last)
                p = p - step_size *  dVdq / (2 * temp)  # half step
            except Exception as e:
                print("passing final of leapfrog due to: {}".format(e))
                q = q_last

            # momentum flip at end
            return q, -p, guess

        def hmc(last_sample, guess):
            p0 = torch.randn(fsm.vector_dim) * std
            q_new, p_new, guess_new = leapfrog(last_sample, p0, guess)
            start_f = fem.f(last_sample, initial_guess=guess)
            start_log_p = - (start_f+EPS)/(temp*sq(last_sample)) - (p0**2).sum() * std**2
            start_log_p = start_log_p.detach().cpu().numpy()
            new_f = fem.f(q_new, initial_guess=guess_new)
            new_log_p = - (new_f+EPS)/(temp*sq(q_new)) - (p_new**2).sum() * std**2
            new_log_p = new_log_p.detach().cpu().numpy()
            if np.isclose(new_log_p, start_log_p) and np.all(np.isclose(q_new.detach().cpu().numpy(), last_sample.detach().cpu().numpy())):
                return q_new, guess_new, torch.zeros(fsm.vector_dim), fa.Function(fsm.V).vector()
            elif np.log(np.random.rand()) < new_log_p - start_log_p:
                return q_new, guess_new, q_new, guess_new
            else:
                return q_new, guess_new, last_sample, guess

        u, guess, new_u, new_guess = hmc(self.last_sample, self.guess)
        
        f, JV, H = fem.f_J_H(u, initial_guess=guess)

        self.last_sample = new_u
        self.guess = new_guess
        p = torch.Tensor([self.args.c1, self.args.c2])
        f = torch.Tensor([f])
        J = self.fsm.to_torch(JV)

        return Example(u, p, f, J, H)


@ray.remote(resources={"WorkerFlags": 0.33})
class HMCCollector(HMCCollectorBase):
    pass

