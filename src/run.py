"""Trains a single model defined by args on data found in args.saved_data_poisson

Logs outputs to TensorBoard"""

import numpy as np
import torch

from copy import deepcopy
import os
import ray

import sys

import traceback

import math

import pdb

from .arguments import parser

# from ..runners.online_trainer import OnlineTrainer
from . import fa_combined as fa
from .pde.metamaterial import Metamaterial
from .nets.feed_forward_net import FeedForwardNet
from .maps.function_space_map import FunctionSpaceMap
from .energy_model.surrogate_energy_model import SurrogateEnergyModel
from .logging.tensorboard_logger import Logger as TFLogger
from .runners.trainer import Trainer
from .runners.collector import PolicyCollector
from .runners.hmc_collector import HMCCollector as Collector
from .runners.nmc_collector import AdversarialCollector
from .runners.deploy_collector import DeployCollector

from .runners.evaluator import CompressionEvaluator as Evaluator
from .runners.harvester import Harvester
from .data.buffer import DataBuffer
from .util.exponential_moving_stats import ExponentialMovingStats

from .util.timer import Timer
import time
import io


def _cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    if args.run_local:
        ray.init(resources={"WorkerFlags": 3}, memory=12e9, object_store_memory=5e9)
        # args.verbose = True
    else:
        ray.init(redis_address="localhost:6379")
    time.sleep(0.1)
    # print("Nodes: ", ray.nodes())
    print("Resources: ", ray.cluster_resources())
    print("Available resources: ", ray.available_resources())
    print("{} nodes".format(len(ray.nodes())))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pde = Metamaterial(args)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    data_dir = os.path.join(args.results_dir, args.data_name)

    if not os.path.exists(data_dir):
        print("Making {}".format(data_dir))
        os.mkdir(data_dir)

    run_number = 0
    while os.path.exists(
        os.path.join(data_dir, "{}_{}".format(args.experiment_name, run_number))
    ):
        run_number = run_number + 1
    out_dir = os.path.join(data_dir, "{}_{}".format(args.experiment_name, run_number))
    os.mkdir(out_dir)
    with open(os.path.join(out_dir, "args.txt"), "w+") as argfile:
        for arg in vars(args):
            argfile.write(arg + ": " + str(getattr(args, arg)) + "\n")

    try:
        fsm = FunctionSpaceMap(
            pde.V, args.bV_dim, cuda=torch.cuda.is_available(), args=args
        )

        net = FeedForwardNet(args, fsm)
        net = _cuda(net)

        surrogate = SurrogateEnergyModel(args, net, fsm)

        tflogger = TFLogger(out_dir)



        if args.load_ckpt_dir is not None:
            print("Reloading checkpoint")
            try:
                ckpt = torch.load(os.path.join(data_dir, args.load_ckpt_dir,
                                               'ckpt.pt'))
            except Exception as e:
                ckpt = torch.load(os.path.join(data_dir, args.load_ckpt_dir,
                                               'last_ckpt.pt'))
            train_data = ckpt['traindata']
            val_data = ckpt['valdata']
            if args.load_net_state:
                net.load_state_dict(ckpt['model_state_dict'])

        else:
            if args.reload_data and os.path.exists(
                os.path.join(data_dir, "initial_datasets.pt")
            ):
                print("Reloading initial data")
                datasets = torch.load(os.path.join(data_dir, "initial_datasets.pt"))

                train_data = datasets["train_data"]
                train_data.data = train_data.data[:args.train_size]
                train_data.memory_size = args.train_size
                train_data.safe_idx = args.n_safe
                val_data = datasets["val_data"]
                val_data.data = val_data.data[:args.val_size]
                val_data.memory_size = args.val_size

                print(
                    "Require initial data with sizes: train {}, val {}".format(
                        args.train_size, args.val_size
                    )
                )

                print(
                    "Found initial data with sizes: train {}, val {}".format(
                        train_data.size(), val_data.size()
                    )
                )

                # if necessary fill Guesses
                small_V_dim = len(fa.Function(fsm.small_V).vector())
                for i in range(len(train_data.data)):
                    if len(train_data.data[i]) == 5:
                        train_data.data[i] = (*train_data.data[i], None)
                    if train_data.data[i][-1] is None:
                        train_data.data[i] = (*train_data.data[i][:-1], torch.zeros(small_V_dim))
                for i in range(len(val_data.data)):
                    if len(val_data.data[i]) == 5:
                        val_data.data[i] = (*val_data.data[i], None)
                    if val_data.data[i][-1] is None:
                        val_data.data[i] = (*val_data.data[i][:-1], torch.zeros(small_V_dim))

            else:
                print("Gathering initial data from scratch")
                # ---------- Start data collection
                train_data = DataBuffer(args.train_size, args.n_safe)
                val_data = DataBuffer(args.val_size)

            if train_data.size() < args.train_size or val_data.size() < args.val_size:
                print("Gathering data to fill train and val buffers")
                val_frac = float(args.val_size) / (args.train_size + args.val_size)

                # Collect initial data
                train_harvester = Harvester(
                    args,
                    lambda x: train_data.feed(x),
                    Collector,
                    int(args.max_collectors * (1.0 - val_frac)),
                )
                print("Train harvester size ", train_harvester.max_workers)
                val_harvester = Harvester(
                    args,
                    lambda x: val_data.feed(x),
                    Collector,
                    int(args.max_collectors * val_frac),
                )
                print("Val harvester size ", val_harvester.max_workers)
                harvested = train_data.size() + val_data.size()
                failed = 0
                with Timer() as htimer:
                    last_save_time = time.time()
                    last_msg_time = time.time()
                    while train_data.size() < len(train_data) or val_data.size() < len(
                        val_data
                    ):
                        if train_data.size() < len(train_data):
                            train_harvester.step()
                        if val_data.size() < len(val_data):
                            val_harvester.step()
                        if time.time() > last_msg_time + 5:
                            last_msg_time = time.time()
                            harvested = train_data.size() + val_data.size()
                            failed = train_harvester.n_death + val_harvester.n_death
                            # print("Nodes: ", ray.nodes())
                            print("Resources: ", ray.cluster_resources())
                            if args.verbose:
                                time.sleep(0.1)
                                print("Available resources: ", ray.available_resources())
                            print("{} nodes".format(len(ray.nodes())))
                            print(
                                "{} train collectors, {} val collectors".format(
                                    len(train_harvester.ids_to_workers),
                                    len(val_harvester.ids_to_workers),
                                )
                            )
                            print(
                                "Harvested {} of {} with {} deaths at time={}s".format(
                                    harvested,
                                    len(train_data) + len(val_data),
                                    failed,
                                    htimer.interval,
                                )
                            )
                            last_error_msg = str(train_harvester.last_error)
                            if len(last_error_msg.split("\n")) > 10:
                                last_error_msg = "\n".join(last_error_msg.split("\n")[-10:])
                            print(
                                "Last error {}s ago: {}".format(
                                    time.time() - train_harvester.last_error_time,
                                    last_error_msg,
                                )
                            )
                            if time.time() > last_save_time + 300:  # Save every 5min
                                print("Saving intermediate data...")
                                last_save_time = time.time()
                                torch.save(
                                    {"train_data": train_data, "val_data": val_data},
                                    os.path.join(data_dir, "initial_datasets.pt"),
                                )
                                print("Saved intermediate data.")

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
                print("Saving collected data...")
                torch.save(
                    {"train_data": train_data, "val_data": val_data},
                    os.path.join(data_dir, "initial_datasets.pt"),
                )
                print("Saved collected data.")

                time.sleep(0.1)


            # ---------- Finish data collection


            if args.adv_collect or args.deploy_collect:
                train_data.memory_size = train_data.memory_size * 5



        def is_valid(example):
            if (
                any([not torch.isfinite(x).all() for x in example])
                or example[2] <= 0
                or example[3].mean().norm() > 1e-7
            ):
                return False
            else:
                return True

        # Filter invalid data
        train_data.data = [d for d in train_data.data if is_valid(d)]
        val_data.data = [d for d in val_data.data if is_valid(d)]

        trainer = Trainer(args, surrogate, train_data, val_data, tflogger, pde)

        if args.load_ckpt_dir is not None and args.load_net_state:
            if args.load_opt_state:
                trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        else:
            # Init whitening module
            preprocd_u = surrogate.net.preproc(
                torch.stack([u for u, _, _, _, _, _ in trainer.train_data.data])
            )

            # surrogate.net.normalizer.mean.data = torch.mean(
            #    preprocd_u, dim=0, keepdims=True
            # ).data

            surrogate.net.normalizer.var.data = (
                torch.std(preprocd_u, dim=0, keepdims=True) ** 2
            ).data

        deploy_ems = ExponentialMovingStats(args.deploy_error_alpha)

        if args.adv_collect and args.deploy_collect:
            raise Exception("Only one of adv_collect and deploy_collect allowed")
        online_harvester = Harvester(
            args,
            lambda x: train_data.feed(x),
            AdversarialCollector if args.adv_collect else DeployCollector,
            args.max_collectors if (args.adv_collect or args.deploy_collect) else 0,
        )

        def deploy_feed(x):
            deploy_ems.feed(x[0])
            # pdb.set_trace()
            img = io.BytesIO(bytes(x[1]))
            tflogger.log_images("Deployment trajectory", [img], x[2])

        deploy_harvester = Harvester(
            args, deploy_feed, Evaluator, args.max_evaluators if args.deploy else 0
        )

        n_batches = len(trainer.train_loader)
        step = 0
        epoch = 0

        ids_to_collectors = {}
        ids_to_evaluators = {}

        last_viz_time = time.time()

        last_state_dict = None
        state_dict = None
        while step < args.max_train_steps:
            # pdb.set_trace()
            # [f_loss, f_pce, J_loss, J_cossim, loss]
            t_losses = np.zeros(7)

            last_state_dict = state_dict
            state_dict = surrogate.net.state_dict()
            state_dict = {
                k: (deepcopy(v).cpu() if hasattr(v, "cpu") else deepcopy(v))
                for k, v in state_dict.items()
            }
            if args.deploy_collect:
                last_state_dict = state_dict # Use most up to date state_dict

            if last_state_dict is not None:
                broadcast_net_state = ray.put(last_state_dict)

            surrogate.net.train()
            train_step_time = 0.0
            for bidx, batch in enumerate(trainer.train_loader):
                if step % 100 == 0:
                    print("step {}, tsize {}".format(step, len(trainer.train_data.data)))
                # if step % 100 == 0:
                #     pdb.set_trace()
                with Timer() as train_step_timer:
                    t_losses += np.array(trainer.train_step(step, batch)) / n_batches
                    if args.cd and (bidx - 1) % args.cd_sgld_steps == 0:
                        trainer.cd_step(step, batch)
                train_step_time += train_step_timer.interval / n_batches

                if last_state_dict is not None and step > 2500:
                    if args.adv_collect:
                        online_harvester.step(init_args=(broadcast_net_state,),
                                              step_args=(batch,))
                    if args.deploy_collect:
                        online_harvester.step(init_args=(broadcast_net_state,),
                                              step_args=())
                    if args.deploy:
                        deploy_harvester.step(step_args=(broadcast_net_state, step))

                step += 1
            # pdb.set_trace()
            epoch += 1

            surrogate.net.eval()

            v_losses = trainer.val_step(step)

            # visualize and save at most once every 2min
            if time.time() - last_viz_time > 120:
                trainer.visualize(
                    step - 1, next(iter(trainer.train_loader)), "Training"
                )
                trainer.visualize(
                    step - 1, next(iter(trainer.val_loader)), "Validation"
                )
                print("Saving checkpoints...")
                if os.path.exists(os.path.join(out_dir, "ckpt.pt")):
                    os.replace(
                        os.path.join(out_dir, "ckpt.pt"),
                        os.path.join(out_dir, "ckpt_last.pt"),
                    )
                torch.save(
                    {
                        "args": args,
                        "epoch": epoch,
                        "traindata": trainer.train_data,
                        "valdata": trainer.val_data,
                        "model_state_dict": surrogate.net.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "tloss": t_losses[6],
                        "vloss": v_losses[6],
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )
                print("Saved checkpoints.")
                last_viz_time = time.time()

            msg = (
                "step {}, epoch {}: "
                "tfL: {:.3e}, tf%: {:.3e}, tJL: {:.3e}, tJsim: {:.3e}, "
                "tHL: {:.3e}, tHsim: {:.3e}, tL: {:.3e}, "
                "vfL: {:.3e}, vf%: {:.3e}, vJL: {:.3e}, vJsim: {:.3e}, "
                "vHL: {:.3e}, vHsim: {:.3e}, vL: {:.3e}, "
                "dloss_mean: {:.3e}, dloss_std: {:.3e}, dloss_90: {:.3e}, "
                "dloss_50: {:.3e}, dloss_10: {:.3e}\n".format(
                    step,
                    epoch,
                    *t_losses,
                    *v_losses,
                    deploy_ems.mean,
                    deploy_ems.std,
                    deploy_ems.m90,
                    deploy_ems.m50,
                    deploy_ems.m10,
                )
            )

            print(msg)
            print("avg_step_time: ", train_step_time)

            with open(os.path.join(out_dir, "losses.txt"), "a") as lossfile:
                lossfile.write(msg)

            print(
                "Harvest stats: onlinesuccess {}, onlinedeath {}, "
                "depsuccess {}, depdeath {}".format(
                    online_harvester.n_success,
                    online_harvester.n_death,
                    deploy_harvester.n_success,
                    deploy_harvester.n_death,
                )
            )
            # print(
            #    "Deploy harvester last error {}s ago: {}".format(
            #        time.time() - deploy_harvester.last_error_time,
            #        deploy_harvester.last_error,
            #    )
            #)
            last_error_msg = str(online_harvester.last_error)
            if len(last_error_msg.split("\n")) > 10:
                last_error_msg = "\n".join(last_error_msg.split("\n")[-10:])
            print(
                "online harvester last error {}s ago: {}".format(
                    time.time() - online_harvester.last_error_time,
                    last_error_msg,
                )
            )
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        with open(os.path.join(out_dir, "exception.txt"), "w") as efile:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=efile)
            raise e
