import torch
from src.arguments import parser
from src import fa_combined as fa
from src.pde.metamaterial import Metamaterial
from src.maps.function_space_map import FunctionSpaceMap
from src.nets.feed_forward_net import FeedForwardNet
from src.energy_model.surrogate_energy_model import SurrogateEnergyModel
from src.energy_model.composed_energy_model import ComposedEnergyModel
from src.energy_model.composed_fenics_energy_model import ComposedFenicsEnergyModel
import matplotlib.pyplot as plt
import numpy as np

load_ckpt_path = "gen_deploy_16"

ckpt = torch.load(
    "/efs_nmor/results/bV_10_hmc_macro/" + load_ckpt_path + "/ckpt_last.pt",
    map_location=torch.device("cpu"),
)

fa.set_log_level(20)

args = parser.parse_args([])

pde = Metamaterial(args)

fsm = FunctionSpaceMap(pde.V, args.bV_dim, cuda=False)
net = FeedForwardNet(args, fsm)
net.load_state_dict(ckpt["model_state_dict"])

Xi1s = [0.0, -0.1, 0.1, -0.2, -0.16, -0.23, -0.4]
Xi2s = [0.0, 0.1, -0.1, 0.07, 0.21, -0.18, -0.06]

RVES_WIDTH = 4

MESH_SIZES = [16, 8, 4, 2, 1]
PORE_RES = [64, 32, 16, 8, 4]

MESH_SIZES = MESH_SIZES[::-1]
PORE_RES = PORE_RES[::-1]

sem = SurrogateEnergyModel(args, net, fsm)
cem = ComposedEnergyModel(args, sem, RVES_WIDTH, RVES_WIDTH)

constrained_sides = [True, False, True, False]
cem_constraint_mask = torch.zeros(len(cem.global_coords))
cem_constraint_mask[cem.bot_idxs()] = 1.0
cem_constraint_mask[cem.top_idxs()] = 1.0
force_data = torch.zeros(len(cem.global_coords), 2)

cem.args.solve_optimizer = "lbfgs"

MAX_DISP = -0.125

args.rtol = 1e-5
args.atol = 1e-5
args.fenics_solver = "newton"

from src.util.timer import Timer

factors = [0.95, 0.9, 0.7, 0.9, 0.7, 0.5, 0.7, 0.5, 0.3, 0.5, 0.3, 0.1, 0.3, 0.1]
anneal_steps = [1, 1, 1, 2, 2, 2, 4, 4, 4, 7, 7, 7, 10, 10]
max_iters = [10, 20, 20, 20, 40, 40, 40, 80, 120, 120, 160, 160, 160, 160]
assert len(factors) == len(anneal_steps)
assert len(factors) == len(max_iters)

base_expr = fa.Expression(("0.0", "X*x[1]"), element=pde.V.ufl_element(), X=MAX_DISP)

for Xi1, Xi2 in zip(Xi1s, Xi2s):
    print("Pores: {}, {}".format(Xi1, Xi2))
    boundary_data = torch.zeros(len(cem.global_coords), 2)
    boundary_data[:, 1] = MAX_DISP * torch.Tensor(cem.global_coords)[:, 1]
    params = torch.zeros(RVES_WIDTH * RVES_WIDTH, 2)
    params[:, 0] = Xi1
    params[:, 1] = Xi2

    with Timer() as t:
        surr_soln, traj_u, traj_f, traj_g = cem.solve(
            params,
            boundary_data,
            cem_constraint_mask,
            force_data,
            step_size=0.2,
            opt_steps=5000,
            return_intermediate=True,
        )
    surr_time = t.interval
    surr_numel = surr_soln.numel()
    surr_E = cem.energy(surr_soln, params, force_data)
    surr_coords = surr_soln.data.cpu().numpy()

    fem_times = []
    fem_Es = []
    fem_numels = []
    fem_solns = []
    fem_coords = []
    fem_fams = []

    for ms, pr in zip(MESH_SIZES, PORE_RES):
        print("{}, {}".format(ms, pr))
        args.composed_mesh_size = ms
        args.composed_pore_resolution = pr


        for idx, (factor, anneal, max_iter) in enumerate(zip(factors, anneal_steps, max_iters)):
            ANNEAL_STEPS = anneal
            args.relaxation_parameter = factor
            args.max_newton_iter = max_iter
            try:
                with Timer() as t:
                    cfem = ComposedFenicsEnergyModel(
                        args,
                        RVES_WIDTH,
                        RVES_WIDTH,
                        Xi1 * np.ones(RVES_WIDTH * RVES_WIDTH),
                        Xi2 * np.ones(RVES_WIDTH * RVES_WIDTH),
                    )
                    print(len(fa.Function(cfem.pde.V).vector()))
                    init_guess = fa.Function(cfem.pde.V).vector()

                    for i in range(ANNEAL_STEPS):
                        print("Anneal {} of {}".format(i + 1, ANNEAL_STEPS))
                        fenics_boundary_fn = fa.Expression(
                            ("0.0", "X*x[1]"),
                            element=pde.V.ufl_element(),
                            X=MAX_DISP * (i + 1) / ANNEAL_STEPS,
                        )
                        true_soln = cfem.solve(
                            args=args,
                            boundary_fn=fenics_boundary_fn,
                            constrained_sides=constrained_sides,
                            initial_guess=init_guess,
                        )
                        init_guess = true_soln.vector()
                break
            except Exception as e:
                print(e)
                if idx >= len(factors) - 1:
                    cfem = ComposedFenicsEnergyModel(
                        args,
                        RVES_WIDTH,
                        RVES_WIDTH,
                        Xi1 * np.ones(RVES_WIDTH * RVES_WIDTH),
                        Xi2 * np.ones(RVES_WIDTH * RVES_WIDTH),
                    )
                    soln = fa.project(base_expr, cfem.pde.V)
    fem_times.append(t.interval)
    print("time ", t.interval)
    print("energy ", cfem.pde.energy(soln))
    fem_Es.append(float(cfem.pde.energy(soln)))
    fem_numels.append(len(fa.Function(cfem.pde.V)))
    fem_solns.append(true_soln)
    initial_coords = np.array(cem.global_coords)
    coords = np.array([true_soln(*x) for x in initial_coords])
    fem_coords.append(coords)
    fem_fams.append((factor, anneal, max_iter))

    torch.save(
        {
            "Xi": (Xi1, Xi2),
            "surr_time": surr_time,
            "surr_numel": surr_numel,
            "surr_E": surr_E,
            "surr_coords": surr_coords,
            "mesh_sizes": MESH_SIZES,
            "pore_resolutions": PORE_RES,
            "fem_times": fem_times,
            "fem_numels": fem_numels,
            "fem_coords": fem_coords,
            "fem_Es": fem_Es,
            "fem_fams": fem_fams,
        },
        "benchmark_Xi1{}_Xi2{}_ckpt_{}".format(Xi1, Xi2, load_ckpt_path),
    )
