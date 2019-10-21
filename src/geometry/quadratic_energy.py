from .. import fa_combined as fa
import torch
import numpy as np
from ..arguments import parser
from ..pde.metamaterial import Metamaterial
from ..maps.function_space_map import FunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
import pdb


def build_quadratic_energy(fem):
    Q = np.zeros([fem.fsm.vector_dim, fem.fsm.vector_dim])
    u0 = np.zeros(fem.fsm.vector_dim)
    for i in range(fem.fsm.vector_dim):
        vector = np.zeros(fem.fsm.vector_dim)
        vector[i] = 1.0
        energy, jac, hvp = fem.f_J_Hvp(u0, vector)
        hvp.is_fa_gradient = True
        hvp.is_fa_hessian = False
        Q[i, :] = fem.fsm.to_numpy(hvp)
    return 0.5 * Q


if __name__ == '__main__':
    fa.set_log_level(20)
    args = parser.parse_args()
    pde = Metamaterial(args)
    fsm = FunctionSpaceMap(pde.V, 5)
    fem = FenicsEnergyModel(args, pde, fsm)
    Q = build_quadratic_energy(fem)
    pdb.set_trace()
