"""Poisson PDE definition"""

from .. import fa_combined as fa
from .pde import PDE


class Poisson(PDE):
    def _build_mesh(self):
        '''Build UnitSquareMesh'''

        mesh = fa.UnitSquareMesh(self.args.poisson_mesh_size,
                                 self.args.poisson_mesh_size)
        self.mesh = mesh

    def _build_function_space(self):
        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (fa.near(x[1], 1.0) or fa.near(
                    x[0], 1.0) or fa.near(x[0], 0) or fa.near(x[1], 0))

        self.exterior = Exterior()
        # FunctionSpace is scalar for Poisson
        self.V = fa.FunctionSpace(self.mesh, "P", 1)

    def _energy_density(self, u):
        # We write Poisson PDE in energy minimization form
        # The usual specification is: u solves - Delta u = p
        # instead we write u = argmin integral E
        # where E = 1/2 <grad(u), grad(u)> - up
        # Ref: Evans Partial differential equations 2nd ed., §2.2.5
        # We set p = -1
        return 0.5 * fa.inner(fa.grad(u), fa.grad(u)) + u
