import numpy as np
from ..pde.metamaterial import Metamaterial
from .random_fourier_fn import make_random_fourier_expression
import torch
import random
import math
from .. import fa_combined as fa


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


def make_compression_deploy_bc(args, cem):

    # MAX_DISP = np.random.uniform(0.2, -0.2)
    MAX_DISP = -0.125
    boundary_data = torch.zeros_like(torch.Tensor(cem.global_coords))

    boundary_data = torch.zeros(len(cem.global_coords), 2)
    boundary_data[:, 1] = MAX_DISP * torch.Tensor(cem.global_coords)[:, 1]

    constrained_sides = [False, False, False, False]
    while not any(constrained_sides):
        constrained_sides = [random.random() < 0.5 for _ in constrained_sides]

    cem_constraint_mask = torch.zeros(len(cem.global_coords))
    cem_constraint_mask[cem.bot_idxs()] = 1.0
    cem_constraint_mask[cem.top_idxs()] = 1.0

    return boundary_data, cem_constraint_mask


def make_random_deploy_bc(args, cem):
    freq_scale = np.random.uniform(0.0, args.boundary_freq_scale)
    amp_scale = np.random.uniform(0.0, args.boundary_amp_scale)
    shear_scale = np.random.uniform(0.0, args.boundary_shear_scale)
    ax_scale = np.random.uniform(0.0, args.boundary_ax_scale)
    boundary_expression = make_random_fourier_expression(
        2, 5000, amp_scale, freq_scale, fsm.V.ufl_element()
    )

    W = np.random.randn(2, 2)

    lin_expression = fa.Expression(
        ("w00*x[0] + w01*x[1]", "w10*x[0]+w11*x[1]"),
        w00=W[0, 0] * ax_scale,
        w01=W[0, 1] * shear_scale,
        w10=W[1, 0] * shear_scale,
        w11=W[1, 1] * ax_scale,
        degree=2,
    )

    boundary_data = torch.zeros_like(torch.Tensor(cem.global_coords))

    for s in range(len(boundary_data)):
        x1, x2 = cem.global_coords[s]
        u1, u2 = boundary_expression([x1, x2])
        u1l, u2l = lin_expression([x1, x2])

        boundary_data[s][0] += u1 + u1l
        boundary_data[s][1] += u2 + u2l

    constrained_sides = [False, False, False, False]
    while not any(constrained_sides):
        constrained_sides = [random.random() < 0.5 for _ in constrained_sides]

    cem_constraint_mask = torch.zeros(len(cem.global_coords))
    if constrained_sides[0]:
        cem_constraint_mask[cem.bot_idxs()] = 1.0
    elif constrained_sides[1]:
        cem_constraint_mask[cem.rhs_idxs()] = 1.0
    elif constrained_sides[2]:
        cem_constraint_mask[cem.top_idxs()] = 1.0
    elif constrained_sides[3]:
        cem_constraint_mask[cem.lhs_idxs()] = 1.0

    return boundary_data, cem_constraint_mask

def make_bc(args, fsm):
    freq_scale = np.random.uniform(0.0, args.boundary_freq_scale)
    amp_scale = np.random.uniform(0.0, args.boundary_amp_scale)
    gauss_scale = np.random.uniform(0.0, args.boundary_gauss_scale)
    sin_scale = np.random.uniform(0.0, args.boundary_sin_scale)
    shear_scale = np.random.uniform(0.0, args.boundary_shear_scale)
    ax_scale = np.random.uniform(0.0, args.boundary_ax_scale)

    boundary_expression = make_random_fourier_expression(
        2, 5000, amp_scale, freq_scale, fsm.V.ufl_element()
    )

    W = np.random.randn(2, 2)

    lin_expression = fa.Expression(
        ("w00*x[0] + w01*x[1]", "w10*x[0]+w11*x[1]"),
        w00=W[0, 0] * ax_scale,
        w01=W[0, 1] * shear_scale,
        w10=W[1, 0] * shear_scale,
        w11=W[1, 1] * ax_scale,
        degree=2,
    )

    sin_expression = fa.Expression(
        ("a*sin(b*x[1]+t)", "-a*sin(b*x[0]+t)"),
        a=sin_scale,
        b=2 * math.pi * (random.random() + 0.5),
        t=2 * math.pi * random.random(),
        degree=2,
    )

    boundary_data = np.random.randn(4 * fsm.elems_along_edge, 2) * gauss_scale

    for s in range(len(boundary_data)):
        if s % fsm.elems_along_edge == 0:
            sin_intensity = random.random()
        x1, x2 = fsm.s_to_x(s)
        u1, u2 = boundary_expression([x1, x2])
        u1s, u2s = np.array(sin_expression([x1, x2])) * sin_intensity
        u1l, u2l = lin_expression([x1, x2])

        boundary_data[s][0] += u1 + u1s + u1l
        boundary_data[s][1] += u2 + u2s + u2l

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

    constraint_mask = fsm.to_torch(constraint_mask)

    return boundary_data, constrained_idxs, constrained_sides, constraint_mask



if __name__ == '__main__':
    from ..nets.feed_forward_net import FeedForwardNet
    from ..maps.function_space_map import FunctionSpaceMap
    from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
    from ..energy_model.composed_energy_model import ComposedEnergyModel
    from ..arguments import parser
    import pdb

    args = parser.parse_args()

    pde = Metamaterial(args)
    fsm = FunctionSpaceMap(pde.V, args.bV_dim)
    net = FeedForwardNet(args, fsm)
    sem = SurrogateEnergyModel(args, net, fsm)
    cem = ComposedEnergyModel(args, sem, 4, 4)

    print(make_compression_deploy_bc(args, cem))
    print(make_random_deploy_bc(args, cem))
