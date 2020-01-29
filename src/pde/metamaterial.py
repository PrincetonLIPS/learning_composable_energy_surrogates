"""Metamaterial PDE definition"""

import math
import numpy as np
import mshr
from .. import fa_combined as fa
from .pde import PDE
from .strain import NeoHookeanEnergy
import pdb


class Metamaterial(PDE):
    def _build_mesh(self):
        """Create mesh with pores defined by c1, c2 a la Overvelde&Bertoldi"""
        args = self.args
        (
            L0,
            porosity,
            c1,
            c2,
            resolution,
            n_cells,
            min_feature_size,
            pore_radial_resolution,
        ) = (
            args.L0,
            args.porosity,
            args.c1,
            args.c2,
            args.metamaterial_mesh_size,
            args.n_cells,
            args.min_feature_size,
            args.pore_radial_resolution,
        )
        if L0 is None:
            L0 = 1.0 / self.args.n_cells

        self.mesh = make_metamaterial_mesh(
            L0,
            c1,
            c2,
            pore_radial_resolution,
            min_feature_size,
            resolution,
            n_cells,
            porosity,
        )

    def _build_function_space(self):
        """Create 2d VectorFunctionSpace and an exterior domain"""
        L0 = self.args.L0
        if L0 is None:
            L0 = 1.0 / self.args.n_cells
        n_cells = self.args.n_cells

        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], L0 * n_cells)
                    or fa.near(x[0], L0 * n_cells)
                    or fa.near(x[0], 0)
                    or fa.near(x[1], 0)
                )

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 0)

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], L0 * n_cells)

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], 0)

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], L0 * n_cells)

        self.exteriors_dic = {
            "left": Left(),
            "right": Right(),
            "bottom": Bottom(),
            "top": Top(),
        }
        self.exterior = Exterior()
        self.V = fa.VectorFunctionSpace(self.mesh, "P", 2)

        self.sub_domains = fa.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        self.sub_domains.set_all(0)

        self.boundaries_id_dic = {"left": 1, "right": 2, "bottom": 3, "top": 4}
        self.left = Left()
        self.left.mark(self.sub_domains, 1)
        self.right = Right()
        self.right.mark(self.sub_domains, 2)
        self.bottom = Bottom()
        self.bottom.mark(self.sub_domains, 3)
        self.top = Top()
        self.top.mark(self.sub_domains, 4)

        self.normal = fa.FacetNormal(self.mesh)

        self.ds = fa.Measure("ds")(subdomain_data=self.sub_domains)

    def _energy_density(self, u, return_stress=False):
        """Energy density is NeoHookean strain energy. See strain.py for def."""
        return NeoHookeanEnergy(
            u, self.args.young_modulus, self.args.poisson_ratio, return_stress
        )


""" Helper functions """


def make_metamaterial_mesh(
    L0, c1, c2, pore_radial_resolution, min_feature_size,
    resolution, n_cells, porosity,
):

    material_domain = None
    base_pore_points = None
    for i in range(n_cells):
        for j in range(n_cells):
            c1_ = c1[i, j] if isinstance(c1, np.ndarray) else c1
            c2_ = c2[i, j] if isinstance(c2, np.ndarray) else c2

            if isinstance(c1, np.ndarray) or base_pore_points is None:
                base_pore_points, radii, thetas = build_base_pore(
                    L0, c1_, c2_, pore_radial_resolution, porosity
                )

                verify_params(base_pore_points, radii, L0, min_feature_size)

            cell = make_cell(i, j, L0, base_pore_points)
            material_domain = (
                cell if material_domain is None else cell + material_domain
            )

    return fa.Mesh(mshr.generate_mesh(material_domain, resolution * n_cells))


def build_base_pore(L0, c1, c2, n_points, porosity):
    # pdb.set_trace()
    r0 = L0 * math.sqrt(2 * porosity) / math.sqrt(math.pi * (2 + c1 ** 2 + c2 ** 2))

    def coords_fn(theta):
        return r0 * (1 + c1 * fa.cos(4 * theta) + c2 * fa.cos(8 * theta))

    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [coords_fn(float(i) * 2 * math.pi / n_points) for i in range(n_points)]
    points = [
        (rtheta * np.cos(theta), rtheta * np.sin(theta))
        for rtheta, theta in zip(radii, thetas)
    ]
    return np.array(points), np.array(radii), np.array(thetas)


def build_pore_polygon(base_pore_points, offset):
    points = [fa.Point(p[0] + offset[0], p[1] + offset[1]) for p in base_pore_points]
    pore = mshr.Polygon(points)
    return pore


def make_cell(i, j, L0, base_pore_points):
    pore = build_pore_polygon(base_pore_points, offset=(L0 * (i + 0.5), L0 * (j + 0.5)))

    cell = mshr.Rectangle(
        fa.Point(L0 * i, L0 * j), fa.Point(L0 * (i + 1), L0 * (j + 1))
    )
    material_in_cell = cell - pore
    return material_in_cell


def verify_params(pore_points, radii, L0, min_feature_size):
    """Verify that params correspond to a geometrically valid structure"""
    # check Constraint A
    tmin = L0 - 2 * pore_points[:, 1].max()
    if tmin / L0 <= min_feature_size:
        raise ValueError(
            "Minimum material thickness violated. Params do not "
            "satisfy Constraint A from Overvelde & Bertoldi"
        )

    # check Constraint B
    # Overvelde & Bertoldi check that min radius > 0.
    # we check it is > min_feature_size > 2.0, so min_feature_size can be used
    # to ensure the material can be fabricated
    if radii.min() <= min_feature_size / 2.0:
        raise ValueError(
            "Minimum pore thickness violated. Params do not "
            "satisfy (our stricter version of) Constraint B "
            "from Overvelde & Bertoldi"
        )


class PoissonMetamaterial(Metamaterial):
    def _energy_density(self, u):
        """Energy density is NeoHookean strain energy. See strain.py for def."""
        f = u - fa.exp(u)
        return 0.5 * fa.inner(fa.grad(u), fa.grad(u)) - f

    def _build_function_space(self):
        super(PoissonMetamaterial, self)._build_function_space()
        self.V = fa.FunctionSpace(self.mesh, 'P', 1)



def make_metamaterial(args, mesh=None):
    if args.poisson:
        return PoissonMetamaterial(args, mesh)
    else:
        return Metamaterial(args, mesh)
