import torch
from .. import fa_combined as fa
from ..pde.metamaterial import make_metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..energy_model.surrogate_energy_model import SurrogateEnergyModel
from ..data.sample_params import make_p, make_bc, make_force
from ..nets.feed_forward_net import FeedForwardNet
from ..viz.plotting import plot_boundary
from ..energy_model.composed_energy_model import ComposedEnergyModel
from ..energy_model.composed_fenics_energy_model import ComposedFenicsEnergyModel

import math
import matplotlib.pyplot as plt
import io
from PIL import Image

import numpy as np
import pdb

import ray

from .evaluator_base import EvaluatorBase, CompressionEvaluatorBase


RVES_WIDTH = 4


@ray.remote(resources={"WorkerFlags": 0.5})
class Evaluator(object):
    pass


@ray.remote(resources={"WorkerFlags": 1.0})
class CompressionEvaluator(CompressionEvaluatorBase):
    pass


if __name__ == "__main__":
    from ..arguments import parser

    args = parser.parse_args()
    fa.set_log_level(20)
    ce = CompressionEvaluatorBase(args, 0)
    ce.step(None, 0)
