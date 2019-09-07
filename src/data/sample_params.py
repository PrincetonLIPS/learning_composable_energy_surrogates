import numpy as np
from ..pde.metamaterial import Metamaterial
from .random_fourier_fn import make_random_fourier_expression
import torch


def make_p(args):
    if args.sample_c:
        tries = 0
        while True and tries < 100:
            args.c1 = np.random.uniform(args.c1_low, args.c1_high)
            args.c2 = np.random.uniform(args.c2_low, args.c2_high)
            try:
                pde = Metamaterial(args)
                break
            except ValueError as e:
                pass
        if tries >= 100:
            raise Exception("Failed to sample parameters")
    else:
        args.c1 = (args.c1_low + args.c1_high) / 2.0
        args.c2 = (args.c2_low + args.c2_high) / 2.0


def make_force(args, fsm):
    freq_scale = np.random.uniform(0.0, args.force_freq_scale)
    amp_scale = np.random.uniform(0.0, args.force_amp_scale)
    force_expression = make_random_fourier_expression(
        2, 5000, amp_scale, freq_scale, fsm.V.ufl_element()
    )
    force_data = np.zeros([4 * fsm.elems_along_edge, 2])

    for s in range(len(force_data)):
        x1, x2 = fsm.s_to_x(s)
        u1, u2 = force_expression([x1, x2])
        force_data[s][0] = u1
        force_data[s][1] = u2
    return fsm.to_torch(torch.Tensor(force_data))


def make_bc(args, fsm):
    freq_scale = np.random.uniform(0.0, args.boundary_freq_scale)
    amp_scale = np.random.uniform(0.0, args.boundary_amp_scale)
    boundary_expression = make_random_fourier_expression(
        2, 5000, amp_scale, freq_scale, fsm.V.ufl_element()
    )
    boundary_data = np.zeros([4 * fsm.elems_along_edge, 2])

    for s in range(len(boundary_data)):
        x1, x2 = fsm.s_to_x(s)
        u1, u2 = boundary_expression([x1, x2])
        boundary_data[s][0] = u1
        boundary_data[s][1] = u2

    boundary_data = fsm.to_torch(torch.Tensor(boundary_data))
    # Randomly constrain sides
    constrained_sides = [True, True, True, True]
    while sum(constrained_sides) == 4 or sum(constrained_sides) == 0:
        constrained_sides = [False, False, False, False]
        for i in range(len(constrained_sides)):
            if np.random.random() < 0.5:
                constrained_sides[i] = True

    constrained_idxs = []
    if constrained_sides[0]:
        constrained_idxs = constrained_idxs + fsm.bottom_idxs()
    if constrained_sides[1]:
        constrained_idxs = constrained_idxs + fsm.rhs_idxs()
    if constrained_sides[2]:
        constrained_idxs = constrained_idxs + fsm.top_idxs()
    if constrained_sides[3]:
        constrained_idxs = constrained_idxs + fsm.lhs_idxs()

    constrained_idxs = sorted(list(set(constrained_idxs)))

    constraint_mask = torch.zeros_like(fsm.to_ring(boundary_data))
    constraint_mask[constrained_idxs, 0] = 1
    constraint_mask[constrained_idxs, 1] = 1

    return boundary_data, constrained_idxs, constrained_sides, constraint_mask
