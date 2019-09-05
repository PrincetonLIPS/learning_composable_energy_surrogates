"""Trains a single model defined by args on data found in args.saved_data_poisson

Logs outputs to TensorBoard"""

import numpy as np
import torch

from copy import deepcopy
import os
import ray

import sys

import traceback

from .arguments import parser

# from ..runners.online_trainer import OnlineTrainer
from .pde.metamaterial import Metamaterial
from .nets.feed_forward_net import FeedForwardNet
from .maps.function_space_map import FunctionSpaceMap
from .energy_model.surrogate_energy_model import SurrogateEnergyModel
from .logging.tensorboard_logger import Logger as TFLogger
from .runners.trainer import Trainer
from .runners.collector import Collector, PolicyCollector
from .runners.evaluator import Evaluator
from .runners.harvester import Harvester
from .data.buffer import DataBuffer
from .util.exponential_moving_stats import ExponentialMovingStats

from .util.timer import Timer


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pde = Metamaterial(args)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    out_dir = os.path.join(args.results_dir, args.experiment_name)

    i = 0
    while os.path.exists(out_dir + "_" + str(i)):
        i += 1
    out_dir = out_dir + "_" + str(i)
    print("Making {}".format(out_dir))
    os.mkdir(out_dir)

    try:
        with open(os.path.join(out_dir, "args.txt"), "w+") as argfile:
            for arg in vars(args):
                argfile.write(arg + ": " + str(getattr(args, arg)) + "\n")
        fsm = FunctionSpaceMap(pde.V, args.bV_dim, cuda=True)

        net = FeedForwardNet(args, fsm)
        net = net.cuda()

        surrogate = SurrogateEnergyModel(args, net, fsm)

        tflogger = TFLogger(out_dir)

        train_data = DataBuffer(args.train_size, args.n_safe)
        val_data = DataBuffer(args.val_size)

        trainer = Trainer(args, surrogate, train_data, val_data, tflogger, pde)

        val_frac = float(args.val_size) / (args.train_size + args.val_size)

        ray.init()

        # Collect initial data
        train_harvester = Harvester(
            args, train_data, Collector, int(args.max_collectors * (1.0 - val_frac))
        )
        val_harvester = Harvester(
            args, val_data, Collector, int(args.max_collectors * val_frac)
        )
        harvested = 0
        with Timer() as htimer:
            while train_data.size() < len(train_data) or val_data.size() < len(
                val_data
            ):
                if train_data.size() < len(train_data):
                    train_harvester.step()
                if val_data.size() < len(val_data):
                    val_harvester.step()
                if train_data.size() + val_data.size() > harvested:
                    harvested = train_data.size() + val_data.size()
                    print(
                        "Harvested {} of {} at time={}s".format(
                            harvested, len(train_data) + len(val_data), htimer.interval
                        )
                    )

        print(
            "Initial harvest took {}s: tsuccess {}, tdeath {}, "
            "vsuccess {}, vdeath {}".format(
                htimer.interval,
                train_harvester.n_success,
                train_harvester.n_death,
                val_harvester.n_success,
                val_harvester.n_death,
            )
        )

        # Garbage collect these harvesters and their workers
        del train_harvester
        del val_harvester

        dagger_harvester = Harvester(
            args, train_data, PolicyCollector, args.max_collectors
        )

        deploy_ems = ExponentialMovingStats(args.deploy_error_alpha)
        deploy_harvester = Harvester(args, deploy_ems, Evaluator, args.max_evaluators)

        n_batches = len(trainer.train_loader)
        step = 0
        epoch = 0

        ids_to_collectors = {}
        ids_to_evaluators = {}

        while step < args.max_train_steps:

            # [f_loss, f_pce, J_loss, J_cossim, loss]
            t_losses = np.zeros(5)

            broadcast_net_state = ray.put(deepcopy(surrogate.net).cpu().state_dict())

            surrogate.net.train()
            for bidx, batch in enumerate(trainer.train_loader):
                t_losses += np.array(trainer.train_step(step, batch)) / n_batches
                if args.visualize_every > 0 and (step - 1) % args.visualize_every == 0:
                    trainer.visualize(step - 1, trainer.train_plot_data, "Training")
                    trainer.visualize(step - 1, trainer.val_plot_data, "Validation")

                dagger_harvester.step(init_args=(broadcast_net_state,))
                deploy_harvester.step(step_args=(broadcast_net_state,))

                step += 1
            epoch += 1

            surrogate.net.eval()
            v_losses = trainer.val_step(step)

            with open(os.path.join(out_dir, "losses.txt"), "a") as lossfile:
                lossfile.write(
                    "step {}, epoch {}: "
                    "tfL: {:.3e}, tf%: {:.3e}, tJL: {:.3e}, tJsim: {:.3e}, tL: {:.3e} "
                    "vfL: {:.3e}, vf%: {:.3e}, vJL: {:.3e}, vJsim: {:.3e}, vL: {:.3e} "
                    "dloss_mean: {}, dloss_std: {}, dloss_90: {}, "
                    "dloss_50: {}, dloss_10: {}\n".format(
                        step,
                        epoch,
                        t_losses[0],
                        t_losses[1],
                        t_losses[2],
                        t_losses[3],
                        t_losses[4],
                        v_losses[0],
                        v_losses[1],
                        v_losses[2],
                        v_losses[3],
                        v_losses[4],
                        deploy_ems.mean,
                        deploy_ems.std,
                        deploy_ems.m90,
                        deploy_ems.m50,
                        deploy_ems.m10,
                    )
                )
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        with open(os.path.join(out_dir, "exception.txt"), "w") as efile:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=efile)
        raise e
