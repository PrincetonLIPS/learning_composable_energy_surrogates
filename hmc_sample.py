import os
from src.viz.plotting import plot_boundary
from src.maps.function_space_map import FunctionSpaceMap
from src.geometry.remove_rigid_body import RigidRemover
from src.pde.metamaterial import Metamaterial
from src.data.sample_params import make_bc
from src import fa_combined as fa

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.arguments import parser
import sys

args = parser.parse_args(["--bV_dim", "10"])

from src.energy_model.fenics_energy_model import FenicsEnergyModel

args.relaxation_parameter = 1.0

pde = Metamaterial(args)

fsm = FunctionSpaceMap(pde.V, args.bV_dim, cuda=False)
fsm2 = FunctionSpaceMap(pde.V, 5, cuda=False)

fem = FenicsEnergyModel(args, pde, fsm)
fem2 = FenicsEnergyModel(args, pde, fsm2)

rigid_remover = RigidRemover(fsm)

import random
import time
import pdb
import os
import copy
import math

EPS = 1e-5

BASE_ITER = 50
BASE_FACTOR = 1.0

save_dir = "hmc_samples_" + str(time.time())

os.mkdir(save_dir)


np.random.seed(args.seed)
torch.manual_seed(args.seed)


VW = torch.randn(2)

HW = torch.randn(2)


def periodic_part_norm(input):

    u = fsm.to_ring(rigid_remover(input))
    top = u[fsm.top_idxs()[::-1]]
    lhs = u[fsm.lhs_idxs()[::-1]]
    bot = u[fsm.bottom_idxs()]
    rhs = u[fsm.rhs_idxs()]

    topshift = top - top.mean(dim=0, keepdims=True)
    botshift = bot - bot.mean(dim=0, keepdims=True)

    lshift = lhs - lhs.mean(dim=0, keepdims=True)
    rshift = rhs - rhs.mean(dim=0, keepdims=True)

    periodic_vert = (topshift + botshift) / 2
    periodic_horiz = (lshift + rshift) / 2

    out = torch.zeros_like(u)

    return 4 * (periodic_vert[:, 1].norm() + periodic_horiz[:, 0].norm())

    # out[fsm.top_idxs()[::-1]] = periodic_vert + top.mean(dim=0, keepdims=True)
    # out[fsm.lhs_idxs()[::-1]] = periodic_horiz + lhs.mean(dim=0, keepdims=True)
    # out[fsm.bottom_idxs()] = periodic_vert + bot.mean(dim=0, keepdims=True)
    # out[fsm.rhs_idxs()] = periodic_horiz + rhs.mean(dim=0, keepdims=True)
    # if fsm.is_ring(input):
    #    return u
    # elif fsm.is_torch(input):
    #     return fsm.to_torch(u)
    # else:
    #    raise Exception()


def sq(q):
    q = fsm.to_ring(rigid_remover(q))
    vert = ((q[fsm.top_idxs()].sum(dim=0) - q[fsm.bottom_idxs()].sum(dim=0)) ** 2).sum()
    horiz = ((q[fsm.lhs_idxs()].sum(dim=0) - q[fsm.rhs_idxs()].sum(dim=0)) ** 2).sum()
    # pdb.set_trace()
    print(
        "vert {}, horiz {}, periodic {}".format(
            vert.item(), horiz.item(), periodic_part_norm(q).item()
        )
    )
    return torch.sqrt(
        EPS ** 2 + vert + horiz
    )  # + periodic_part_norm(q)  # + vert + horiz + q.norm() # torch.nn.functional.softplus(vert + horiz) # + (q**2).sum()


def dsq(q):
    q = torch.autograd.Variable(q.data, requires_grad=True)
    return torch.autograd.grad(sq(q), q)[0]


def solve(
    q, guess, q_last=None, max_iter=BASE_ITER, factor=BASE_FACTOR, recursion_depth=0
):
    try:
        print(
            "recursion {}, iter {}, factor {}".format(recursion_depth, max_iter, factor)
        )
        new_args = copy.deepcopy(args)
        new_args = new_args
        new_args.max_newton_iter = max_iter
        new_args.relaxation_parameter = factor
        # new_args.atol = math.sqrt(args.atol)
        # new_args.rtol = math.sqrt(args.rtol)
        # f, u = fem.f(q, initial_guess=guess, return_u=True, args=new_args)
        # guess = u.vector()
        T = 8
        Z = sum([2 ** i for i in range(T)])
        for i in range(T):
            new_args.atol = args.atol  # 10**(math.log10(args.atol)*2**i / (2**(T-1)))
            new_args.rtol = 10 ** (math.log10(args.rtol) * 2 ** i / Z)
            new_args.max_newton_iter = int(math.ceil(2 ** i * max_iter / Z)) + 1
            f, u = fem.f(q, initial_guess=guess, return_u=True, args=new_args)
            guess = u.vector()
        print(
            "energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}, logp_macro: {:.3e}, V: {:.3e}".format(
                f, sq(q), (f + EPS) / sq(q), logp_macro(q), f + logp_macro(q)
            )
        )
        return u.vector()
    except Exception as e:
        if q_last is None:
            raise e
        elif recursion_depth >= 3:
            print("Maximum recursion depth exceeded! giving up.")
            raise e
        else:
            print("recursing due to error:")
            print(e)
            # q_mid = q_last + 0.5*(q-q_last)
            new_factor = factor * 0.3
            new_max_iter = int(
                1
                + max_iter
                * math.log(1.0 - min(0.9, factor))
                / math.log(1.0 - new_factor)
            )
            # guess = solve(q_mid, guess, q_last, max_iter=new_max_iter,
            #               factor=new_factor, recursion_depth=recursion_depth+1)
            return solve(
                q,
                guess,
                q_last,
                max_iter=new_max_iter,
                factor=new_factor,
                recursion_depth=recursion_depth + 1,
            )


macro = torch.Tensor([[0.0, -0.12], [0.0, 0.0]])  # torch.randn(2,2)


def logp_macro(q):
    # Negative log probability
    q = fsm.to_ring(rigid_remover(q))
    vert = q[fsm.top_idxs()].mean(dim=0) - q[fsm.bottom_idxs()].mean(dim=0)
    horiz = q[fsm.lhs_idxs()].mean(dim=0) - q[fsm.rhs_idxs()].mean(dim=0)
    # pdb.set_trace()
    # print("vert {}, horiz {}, periodic {}".format(vert.item(), horiz.item(), periodic_part_norm(q).item()))
    x = torch.stack([vert, horiz]).view(-1)
    mu = macro.view(-1)
    Sigma_inv = mu ** 2
    print(
        "x_macro: {}, residual: {}, scaled_res: {}".format(
            x.data.cpu().numpy(),
            (x - mu).data.cpu().numpy(),
            (Sigma_inv * (x - mu) ** 2).data.cpu().numpy(),
        )
    )
    # pdb.set_trace()
    return 100 * (Sigma_inv * (x - mu) ** 2).sum()


def dlogp_macro(q):
    q = torch.autograd.Variable(q.data, requires_grad=True)
    return torch.autograd.grad(logp_macro(q), q)[0]


def make_V(q, guess, q_last=None):
    guess = solve(q, guess, q_last)
    f, u = fem.f(q, initial_guess=guess, return_u=True)
    return f + logp_macro(q)


def make_dVdq(q, guess, q_last=None):
    guess = solve(q, guess, q_last)
    f, JV, u = fem.f_J(q, initial_guess=guess, return_u=True)
    J = fsm.to_torch(JV)
    # (f'g - g'f)/g^2
    dVdq = J + dlogp_macro(q)
    return dVdq, u.vector()


"""
def make_dVdq(q, guess, q_last=None):
    guess = solve(q, guess, q_last)
    f, JV, u = fem.f_J(q, initial_guess=guess, return_u=True)
    J = fsm.to_torch(JV)
    dVdq = (J * sq(q) - (f + EPS) * 2 * dsq(q)) / sq(q) ** 2
    return dVdq, u.vector()
"""


def visualize(q, u):
    plt.figure(figsize=(5, 5))
    plot_boundary(
        lambda x: (0, 0), 200, label="reference", color="k",
    )
    plot_boundary(
        fsm.get_query_fn(q), 200, label="ub", linestyle="-", color="darkorange",
    )
    plot_boundary(
        fsm.get_query_fn(rigid_remover(q.unsqueeze(0)).squeeze(0)),
        200,
        label="rigid removed",
        linestyle="--",
        color="blue",
    )
    coords = fsm.ring_coords.data.cpu().numpy() + fsm.to_ring(u).data.cpu().numpy()
    plt.scatter(coords[:, 0], coords[:, 1], color="k", label="control points")
    fa.plot(u, mode="displacement")
    plt.legend()
    f = fem.f(q, initial_guess=u.vector())
    plt.title("f: {:.3e}, sq: {:.3e}, f/sq: {:.3e}".format(f, sq(q), (f + EPS) / sq(q)))
    plt.savefig(save_dir + "/hmc_" + str(time.time()) + ".png")


def leapfrog(q, p, guess, path_len, step_size, temp):
    q = q.clone()
    p = p.clone()
    dVdq, guess = make_dVdq(q, guess, q_last=None)
    p = p - step_size * dVdq / (2 * temp)  # half step
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
        p = p - step_size * dVdq / (2 * temp)  # half step
    except Exception as e:
        print("passing final of leapfrog due to: {}".format(e))
        q = q_last

    # momentum flip at end
    return q, -p, guess


def make_dpoint(ub, initial_guess):
    f, J, H = fem.f_J_H(ub, initial_guess=initial_guess)
    return (ub, None, f, fsm.to_torch(J), H)


def hmc(n_samples, path_len=1.0, step_size=0.01, std=0.1, temp=1.0):
    samples = [torch.zeros(fsm.vector_dim)]

    guess = fa.Function(fsm.V).vector()
    guess.set_local(np.random.randn(len(guess)) * 1e-10)
    guesses = [guess]

    start_f = fem.f(samples[-1], initial_guess=guess)
    print("start f: ", start_f)

    last_sample = samples[-1]

    while len(samples) < n_samples + 1:
        p0 = torch.randn(fsm.vector_dim) * std
        try:
            start_f = fem.f(last_sample, initial_guess=guess)
            q_new, p_new, guess_new = leapfrog(
                last_sample, p0, guess, path_len, step_size, temp
            )
            print("start f: ", start_f)
            print("start sq: ", sq(last_sample))
            print("start_logp_macro: ", logp_macro(last_sample))
            start_log_p = (
                -make_V(last_sample, guess) / temp - (p0 ** 2).sum() * std ** 2
            )
            # start_log_p = (
            #    -(start_f + EPS) / (temp * sq(last_sample)) - (p0 ** 2).sum() * std ** 2
            # )
            start_log_p = start_log_p.detach().cpu().numpy()
            print("start log p: ", start_log_p)
            new_f = fem.f(q_new, initial_guess=guess_new)
            print("new f: ", new_f)
            # new_log_p = -(new_f + EPS) / (temp * sq(q_new)) - (p_new ** 2).sum() * std ** 2
            new_log_p = -make_V(q_new, guess_new) / temp - (p0 ** 2).sum() * std ** 2
            new_log_p = new_log_p.detach().cpu().numpy()
            print("new sq: ", sq(q_new))
            print("new_logp_macro: ", logp_macro(q_new))
            print("new log p: ", new_log_p)
            u = fa.Function(fsm.V)
            u.vector().set_local(guess_new)
            visualize(q_new, u)
            if np.isclose(new_log_p, start_log_p) and np.all(
                np.isclose(
                    q_new.detach().cpu().numpy(), samples[-1].detach().cpu().numpy()
                )
            ):
                print("sample rejected due to repetition")
                guess = fa.Function(fsm.V).vector()
                guess.set_local(np.random.randn(len(guess)) * 1e-10)
                last_sample = torch.zeros(fsm.vector_dim)
                # last_sample = samples[-min(2, len(samples))].detach().clone()
                # guess = fa.Function(fsm.V).vector()
            elif np.log(np.random.rand()) < new_log_p - start_log_p:
                samples.append(q_new)
                last_sample = q_new
                guess = guess_new
                guesses.append(guess_new)
                print("sample accepted")
            else:
                samples.append(samples[-1].detach().clone())
                guesses.append(guesses[-1])
                last_sample = samples[-1].detach().clone()
                print("sample rejected")
        except Exception as e:
            print("Failed at start and reseeding due to: {}".format(e))

            guess = fa.Function(fsm.V).vector()
            guess.set_local(np.random.randn(len(guess)) * 1e-10)
            last_sample = torch.zeros(fsm.vector_dim)

    return samples[1:]


fa.set_log_level(20)

samples = hmc(100, path_len=0.2, step_size=0.01, std=0.2, temp=0.005)
pdb.set_trace()
