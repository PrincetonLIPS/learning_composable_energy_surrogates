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

load_ckpt_path = "swa_reload_0"

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

Xi1s = [0.0, -0.0576, 0.0242, 0.0048, -0.0614, -0.0576, -0.184, -0.207]
Xi2s = [0.0, -0.0379, -0.0153, -0.0655, -0.0228, -0.0379, -0.106, 0.121]

RVES_WIDTH = 4

MESH_SIZES = [16, 16, 8, 4, 2, 1]
PORE_RES = [128, 64, 32, 16, 8, 4]

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

factors = [0.9, 0.7, 0.4, 0.1, 0.05]
anneal_steps = [1, 2, 5, 10, 20]
max_iters = [10, 20, 40, 160, 320]
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
            step_size=0.25,
            opt_steps=10000,
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
    fem_figs = []

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
                    true_soln = fa.project(base_expr, cfem.pde.V)
        fem_times.append(t.interval)
        print("time ", t.interval)
        print("energy ", cfem.pde.energy(true_soln))
        fem_Es.append(float(cfem.pde.energy(true_soln)))
        fem_numels.append(len(fa.Function(cfem.pde.V)))
        fem_solns.append(true_soln)
        initial_coords = np.array(cem.global_coords)
        coords = np.array([true_soln(*x) for x in initial_coords])
        prj_fn = fa.project(base_expr, cfem.pde.V)
        prj_coords = np.array([prj_fn(*x) for x in initial_coords])
        fn = true_soln - fa.Constant((prj_coords.mean(axis=0)[0], prj_coords.mean(axis=0)[1]))
        fn = fa.project(fn, cfem.pde.V)
        plt.figure()
        fa.plot(fn, mode='displacement')
        fig = plt.gcf()
        fem_figs.append(fig)
        fem_coords.append(coords)
        fem_fams.append((factor, anneal, max_iter))
    
    torch.save(
        {
            "homogenized_disp": boundary_data,
            "initial_coords": np.array(cem.global_coords),
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
            "fem_figs": fem_figs
        },
        "benchmark_Xi1{}_Xi2{}_ckpt_{}.pt".format(Xi1, Xi2, load_ckpt_path),
    )
