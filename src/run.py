"""Trains a single model defined by args on data found in args.saved_data_poisson

Logs outputs to TensorBoard"""

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

import pdb
from copy import deepcopy
import glob
import os
import ray

import sys

import traceback

from .. import arguments
from .. import fa_combined as fa

# from ..runners.online_trainer import OnlineTrainer
from ..pde.metamaterial import Metamaterial
from ..nets.feed_forward_net import FeedForwardNet
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..logging.tensorboard_logger import Logger as TFLogger
from ..runners.trainer import Trainer
from ..runners.collector import Collector, PolicyCollector
from ..runners.evaluator import Evaluator
from ..util.carefully_get import carefully_get
from ..data.example import Example
from ..data.buffer import DataBuffer
from ..geometry.polar import SemiPolarizer
from ..geometry.remove_rigid_body import RigidRemover


def collect_initial_data(args, train_data):
    ids_to_collectors = {}
    while train_data.size() < len(train_data):

        while len(ids_to_collectors) < args.max_collectors:
            new_collector = Collector.remote(args)
            ids_to_collectors[new_collector.step.remote()] = new_collector

        examples, ids_to_collectors = harvest(ids_to_collectors)
        for e in examples:
            train_data.feed(e)


def policy_collector_step(args, train_data, ids_to_collectors, surrogate):
    if len(ids_to_collectors) < args.max_collectors:
        surrogate = ray.put(surrogate)
    while len(ids_to_collectors) < args.max_collectors:
        new_collector = PolicyCollector.remote(args, surrogate)
        ids_to_collectors[new_collector.step.remote()] = new_collector

    examples, ids_to_collectors = harvest(ids_to_collectors)
    for e in examples:
        train_data.feed(e)

    return ids_to_collectors


def deploy_step(args, ids_to_evaluators, surrogate):
    if len(ids_to_evaluators) < args.max_evaluators:
        surrogate = ray.put(surrogate)
    while len(ids_to_evaluators) < args.max_evaluators:
        new_evaluator = Evaluator.remote(args, surrogate)
        ids_to_evaluators[new_evaluator.step.remote()] = new_evaluator

    deploy_losses, ids_to_evaluators = harvest(ids_to_evaluators, surrogate)

    return deploy_losses, ids_to_evaluators


def harvest(ids_to_workers, *step_args, **step_kwargs):
    ready_ids, remaining_ids = ray.wait(
        [id for id in ids_to_workers.keys()], timeout=1
    )
    results = {id: carefully_get(id) for id in ready_ids}
    valid_results = []
    # Restart or kill workers as necessary
    for id, result in results.items:
        worker = ids_to_workers.pop(id)
        if not isinstance(result, Exception):
            ids_to_workers[worker.step.remote(*step_args, **step_kwargs)] = worker
            valid_results.append(result)

    return valid_results, ids_to_workers


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    args = arguments.args
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
        fsm = FunctionSpaceMap(pde.V, args.elems_along_edge)
        preproc_fns = []
        if args.remove_rigid:
            preproc_fns.append(RigidRemover(fsm))
        if args.semipolarize:
            preproc_fns.append(SemiPolarizer(fsm))

        def preproc(x):
            for fn in preproc_fns:
                x = fn(x)
            return x

        net = FeedForwardNet(fsm.vector_dim + 2, args, preproc=preproc)

        net = net.cuda()

        sem = SurrogateEnergyModel(args, net, fsm)

        tflogger = TFLogger(out_dir)

        train_data = DataBuffer(args.train_size, args.n_safe)
        val_data = DataBuffer(args.val_size)

        trainer = Trainer(args, sem, train_data, val_data, tflogger, pde)

        collect_initial_data(args, train_data)

        # --------------------------------------------
        # I HAVE WRITTEN UP TO HERE
        # AFTER THIS IS TO-MODIFY
        # --------------------------------------------

        for step in range(1, args.max_train_steps):
            train_loss = trainer.train_step(step)
            val_loss = trainer.val_step(step)
            if args.visualize_every > 0 and (step - 1) % args.visualize_every == 0:
                trainer.visualize(step - 1, trainer.train_plot_data, "Training")
                trainer.visualize(step - 1, trainer.val_plot_data, "Validation")
            with open(os.path.join(out_dir, "losses.txt"), "a") as lossfile:
                if evaluator is not None and evaluator.normalizer > 0:
                    deploy_loss = evaluator.running_error / evaluator.normalizer
                else:
                    deploy_loss = None
                lossfile.write(
                    "{}: tloss: {}, vloss: {}, dloss: {}\n".format(
                        step, train_loss, val_loss, deploy_loss
                    )
                )
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        with open(os.path.join(out_dir, "exception.txt"), "w") as efile:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=efile)
        raise e
