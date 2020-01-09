import numpy as np
import torch
from torch.autograd import Variable

from .. import fa_combined as fa
from ..splines.piecewise_spline import make_piecewise_spline_map
from ..pde.metamaterial import Metamaterial


class FunctionSpaceMap(object):
    """Map between V, torch, numpy representations.

    Defines transforms for the chain:
        V <-> ring <-> torch <-> numpy

    Performs type checking, so user can just call ".to_torch(unknown_fn_or_vec)"

    Throw errors if things don't belong to the correct fn space / shape
    """

    def __init__(self, V, bV_dim, cuda=False, args=None):
        self.V = V
        if V.mesh().geometric_dimension() == 2:
            self.mesh = fa.UnitSquareMesh(bV_dim - 1, bV_dim - 1)
            self.bmesh = fa.BoundaryMesh(
                self.mesh, "exterior"
            )
        else:
            raise Exception("Invalid geometric dimension")
        if V.ufl_element().value_size() == 1:
            self.small_V = fa.FunctionSpace(self.mesh, "P", 1)
            self.bV = fa.FunctionSpace(self.bmesh, "P", 1)
        else:
            self.small_V = fa.VectorFunctionSpace(
                self.mesh, "P", 1, dim=V.ufl_element().value_size())
            self.bV = fa.VectorFunctionSpace(
                self.bmesh, "P", 1, dim=V.ufl_element().value_size()
            )
        self.bV_dim = bV_dim
        self.elems_along_edge = self.bV_dim - 1
        self.vector_dim = len(fa.Function(self.bV).vector())
        self.coordinates = self.bV.tabulate_dof_coordinates()
        self.cuda = cuda
        self.create_boundary_measure()

        self.init_ring(V)

        self.A_cpu = self.make_A(V)
        if self.cuda:
            self.A_cuda = self.A_cpu.cuda()

        self.small_A_cpu = self.make_A(self.small_V)
        if self.cuda:
            self.small_A_cuda = self.small_A_cpu.cuda()

        if args is not None:
            self.small_pde = Metamaterial(args, self.mesh)


    def init_ring(self, V):
        self.channels = V.ufl_element().value_size()

        self.ring_coords = torch.zeros(4 * self.elems_along_edge, 2)

        for idx in (
            self.bV.sub(0).dofmap().dofs()
            if self.channels > 1
            else self.bV.dofmap().dofs()
        ):
            x1, x2 = self.coordinates[idx]
            s = int(round(self.x_to_s(x1, x2)))
            self.ring_coords.data[s, 0] = x1
            self.ring_coords.data[s, 1] = x2

        self.vec_to_ring_map_cpu = torch.zeros(
            self.vector_dim, 4 * self.elems_along_edge, 2
        )
        for c in range(self.channels):
            for idx in (
                self.bV.sub(c).dofmap().dofs()
                if self.channels > 1
                else self.bV.dofmap().dofs()
            ):
                x1, x2 = self.coordinates[idx]
                s = int(round(self.x_to_s(x1, x2)))
                self.vec_to_ring_map_cpu[idx, s, c] = 1.0

        self.vec_to_ring_map_cpu = self.vec_to_ring_map_cpu.view(
            self.vector_dim, self.vector_dim
        )
        if self.cuda:
            self.vec_to_ring_map_cuda = self.vec_to_ring_map_cpu.cuda()

    @property
    def A(self):
        return self.A_cuda if self.cuda else self.A_cpu

    @property
    def small_A(self):
        return self.small_A_cuda if self.cuda else self.small_A_cpu

    @property
    def vec_to_ring_map(self):
        if self.cuda:
            return self.vec_to_ring_map_cuda
        else:
            return self.vec_to_ring_map_cpu

    def make_A(self, V):
        coords = np.array(V.tabulate_dof_coordinates()[V.sub(0).dofmap().dofs()])
        # Find s, which we will use to sort coords
        coords_s = np.array([self.x_to_s(x1, x2) for x1, x2 in coords])

        xrefs = [np.array(self.s_to_x(self.x_to_s(x1, x2))) for x1, x2 in coords]

        def radius(x1, x2):
            return np.linalg.norm(np.array([x1, x2]) - 0.5)

        ratios = np.array(
            [radius(*coords[i]) / radius(*xrefs[i]) for i in range(len(coords))]
        )

        svals = np.array([float(i) for i in range(4 * self.elems_along_edge)])

        A = make_piecewise_spline_map(coords_s, len(svals))

        return torch.Tensor(A).float() * torch.Tensor(ratios).view(-1, 1)

    def create_boundary_measure(self):
        mesh = self.V.mesh()

        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], 1.0)
                    or fa.near(x[0], 1.0)
                    or fa.near(x[0], 0)
                    or fa.near(x[1], 0)
                )

        exterior_domain = fa.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        exterior_domain.set_all(0)
        Exterior().mark(exterior_domain, 1)
        self.boundary_ds = fa.Measure("ds")(subdomain_data=exterior_domain)(1)

    def x_to_s(self, x1, x2):
        """It's important not to int-ize anything in here,
        in case we use s in the Spline subclass as a Spline input
        and wish to have arbitrary s"""

        # project to boundary
        d = np.array([x1, x2]) - 0.5
        if x1 == 0.5 and x2 == 0.5:
            # np.abs(d).max() == 0.
            return 0
        scale = 0.5 / np.abs(d).max()
        x1 = (x1 - 0.5) * scale + 0.5
        x2 = (x2 - 0.5) * scale + 0.5

        assert (
            x1 <= fa.DOLFIN_EPS
            or x1 >= 1.0 - fa.DOLFIN_EPS
            or x2 <= fa.DOLFIN_EPS
            or x2 >= 1.0 - fa.DOLFIN_EPS
        )
        assert (
            x1 >= -fa.DOLFIN_EPS
            and x1 <= 1.0 + fa.DOLFIN_EPS
            and x2 >= -fa.DOLFIN_EPS
            and x2 <= 1.0 + fa.DOLFIN_EPS
        )
        # Check which face
        # Bottom
        if x2 <= fa.DOLFIN_EPS:
            return x1 * self.elems_along_edge
        # RHS
        elif x1 >= 1.0 - fa.DOLFIN_EPS:
            return self.elems_along_edge + x2 * self.elems_along_edge
        elif x2 >= 1.0 - fa.DOLFIN_EPS:
            return 2 * self.elems_along_edge + (1.0 - x1) * self.elems_along_edge
        else:
            return 3 * self.elems_along_edge + (1.0 - x2) * self.elems_along_edge

    def s_to_x(self, s):
        """Apparently this isn't used."""
        s = s % (self.elems_along_edge * 4)
        # Bottom
        if s <= self.elems_along_edge:
            return s / self.elems_along_edge, 0.0
        # RHS
        elif s <= 2 * self.elems_along_edge:
            return 1.0, (s - self.elems_along_edge) / self.elems_along_edge
        # Top
        elif s <= 3 * self.elems_along_edge:
            return (3 * self.elems_along_edge - s) / self.elems_along_edge, 1.0
        else:
            return 0.0, (4 * self.elems_along_edge - s) / self.elems_along_edge

    def bottom_idxs(self):
        return [s for s in range(0, self.elems_along_edge + 1)]

    def rhs_idxs(self):
        return [s for s in range(self.elems_along_edge, 2 * self.elems_along_edge + 1)]

    def top_idxs(self):
        return [
            s for s in range(2 * self.elems_along_edge, 3 * self.elems_along_edge + 1)
        ]

    def lhs_idxs(self):
        return [
            s for s in range(3 * self.elems_along_edge, 4 * self.elems_along_edge)
        ] + [0]

    def get_query_fn(self, fn_or_vec):
        # Caching for speed
        fn_or_vec = self.to_V(fn_or_vec)

        def query_fn(x):
            return fn_or_vec(x)

        return query_fn

    def maybe_interpolate(self, fn_or_vec):
        if self.is_in_spaces(fn_or_vec):
            return fn_or_vec
        elif (
            isinstance(fn_or_vec, fa.Expression)
            or isinstance(fn_or_vec, fa.UserExpression)
            or isinstance(fn_or_vec, fa.Constant)
        ):
            return fa.interpolate(fn_or_vec, self.V)
        else:
            raise Exception(
                "fn_or_vec is not in spaces, or Expression, or Constant. "
                "Found {}".format(fn_or_vec)
            )

    def to_V(self, fn_or_vec):
        fn_or_vec = self.maybe_interpolate(fn_or_vec)
        if self.is_V(fn_or_vec):
            return fn_or_vec
        else:
            return self.ring_to_V(self.to_ring(fn_or_vec))

    def to_small_V(self, fn_or_vec):
        fn_or_vec = self.maybe_interpolate(fn_or_vec)
        if self.is_small_V(fn_or_vec):
            return fn_or_vec
        else:
            return self.ring_to_small_V(self.to_ring(fn_or_vec))

    def to_ring(self, fn_or_vec, keep_grad=False):
        fn_or_vec = self.maybe_interpolate(fn_or_vec)
        if self.is_V(fn_or_vec):
            return self.V_to_ring(fn_or_vec)
        elif self.is_ring(fn_or_vec):
            return self._cuda(fn_or_vec)
        else:
            return self.torch_to_ring(self.to_torch(fn_or_vec, keep_grad), keep_grad)

    def to_torch(self, fn_or_vec, keep_grad=False):
        fn_or_vec = self.maybe_interpolate(fn_or_vec)
        if self.is_numpy(fn_or_vec):
            return self.numpy_to_torch(fn_or_vec)
        elif self.is_torch(fn_or_vec):
            return self._cuda(fn_or_vec)
        else:
            return self.ring_to_torch(self.to_ring(fn_or_vec, keep_grad), keep_grad)

    def to_numpy(self, fn_or_vec):
        fn_or_vec = self.maybe_interpolate(fn_or_vec)
        if self.is_numpy(fn_or_vec):
            return fn_or_vec
        else:
            return self.torch_to_numpy(self.to_torch(fn_or_vec))

    def is_V(self, fn_or_vec):
        if not isinstance(fn_or_vec, fa.Function):
            return False
        else:
            return fn_or_vec.function_space() == self.V

    def is_small_V(self, fn_or_vec):
        if not isinstance(fn_or_vec, fa.Function):
            return False
        else:
            return fn_or_vec.function_space() == self.small_V

    def is_numpy(self, fn_or_vec):
        if not isinstance(fn_or_vec, np.ndarray):
            return False
        elif len(fn_or_vec.shape) == 1 and fn_or_vec.shape[0] == self.vector_dim:
            return True
        elif len(fn_or_vec.shape) == 2 and fn_or_vec.shape[1] == self.vector_dim:
            return True
        else:
            raise Exception(
                "Invalid size {}, expected either [{}] or "
                "[batch_size, {}]".format(
                    list(fn_or_vec.shape), self.vector_dim, self.vector_dim
                )
            )

    def is_ring(self, fn_or_vec):
        if not isinstance(fn_or_vec, torch.Tensor):
            return False
        elif len(fn_or_vec.size()) < 2 or len(fn_or_vec.size()) > 3:
            # Expect [n_locs, dim] or [batch, n_locs, dim]
            return False
        elif fn_or_vec.size()[-2] != 4 * self.elems_along_edge:
            return False
        elif fn_or_vec.size()[-1] != self.V.ufl_element().value_size():
            return False
        else:
            return True

    def is_torch(self, fn_or_vec):
        if not isinstance(fn_or_vec, torch.Tensor):
            return False
        elif self.is_ring(fn_or_vec):
            return False
        elif len(fn_or_vec.size()) == 1 and fn_or_vec.size()[0] == self.vector_dim:
            return True
        elif len(fn_or_vec.size()) == 2 and fn_or_vec.size()[1] == self.vector_dim:
            return True
        else:
            raise Exception(
                "Invalid size {}, expected either [{}] or "
                "[batch_size, {}]".format(
                    list(fn_or_vec.size()), self.vector_dim, self.vector_dim
                )
            )

    def is_force(self, fn):
        return self.is_ring(fn)

    def V_to_ring(self, fn_V):
        if not getattr(fn_V, "is_fa_gradient", False) and not getattr(
            fn_V, "is_fa_hessian", False
        ):
            print("Warning: interpolating to ring. "
                  "High-freq information may be lost.")
            x = self.to_ring(torch.zeros([self.vector_dim]))
            for i, xloc in enumerate(self.ring_coords):
                x1, x2 = xloc.data.cpu().numpy()
                y1, y2 = fn_V([x1, x2])
                x[i, 0] = y1
                x[i, 1] = y2
            return x

        if getattr(fn_V, "is_fa_gradient", False):
            return self.V_gradient_to_ring(fn_V)

        elif getattr(fn_V, "is_fa_hessian", False):
            raise Exception("Not implemented!")

        else:
            raise Exception("Shouldn't reach here")

    def V_gradient_to_ring(self, fn_V):
        uvec = self._cuda(torch.Tensor(np.array(fn_V.vector())))  # (|Y|xD)
        uvec = torch.stack(de_interleave(uvec))  # D x |Y|
        grad_ring = torch.matmul(uvec, self.A)  # D x |y|
        grad_ring = grad_ring.transpose(1, 0)  # |y| x D
        return grad_ring

    def ring_to_V(self, fn_ring):
        """fn_ring is |y| x D"""
        assert self.is_ring(fn_ring)
        if len(fn_ring.shape) == 3:
            assert fn_ring.size(0) == 1
            fn_ring = fn_ring.squeeze(0)
        uvec = torch.matmul(self.A, fn_ring)  # |Y| x D
        uvec = interleave(uvec[:, 0], uvec[:, 1])  # (|Y|xD)
        uvec = uvec.data.cpu().numpy()
        fn_V = fa.Function(self.V)
        fn_V.vector().set_local(uvec)
        return fn_V

    def ring_to_small_V(self, fn_ring):
        """fn_ring is |y| x D"""
        assert self.is_ring(fn_ring)
        if len(fn_ring.shape) == 3:
            assert fn_ring.size(0) == 1
            fn_ring = fn_ring.squeeze(0)
        uvec = torch.matmul(self.small_A, fn_ring)  # |Y| x D
        uvec = interleave(uvec[:, 0], uvec[:, 1])  # (|Y|xD)
        uvec = uvec.data.cpu().numpy()
        fn_sV = fa.Function(self.small_V)
        fn_sV.vector().set_local(uvec)
        return fn_sV

    def numpy_to_torch(self, fn_numpy):
        assert self.is_numpy(fn_numpy)
        return Variable(self._cuda(torch.Tensor(fn_numpy)), requires_grad=True)

    def torch_to_numpy(self, fn_torch):
        assert self.is_torch(fn_torch)
        return fn_torch.data.cpu().numpy()

    def torch_to_ring(self, fn_or_vec, keep_grad=False):
        assert self.is_torch(fn_or_vec)
        if len(fn_or_vec.size()) == 1:
            ret = self.torch_to_ring(fn_or_vec.unsqueeze(0), keep_grad=True)
            ring = ret.squeeze(0)
        else:
            ringvec = torch.matmul(fn_or_vec, self.vec_to_ring_map)
            ring = ringvec.view(-1, 4 * self.elems_along_edge, 2)
        if not keep_grad:
            ring = self.proc_torch(ring)
        return ring

    def ring_to_torch(self, fn_or_vec, keep_grad=False):
        assert self.is_ring(fn_or_vec)
        if len(fn_or_vec.size()) == 2:
            vec = self.ring_to_torch(fn_or_vec.unsqueeze(0), keep_grad=True)
            vec = vec.squeeze(0)
        else:
            ringvec = fn_or_vec.reshape(-1, self.vector_dim)
            vec = torch.matmul(ringvec, self.vec_to_ring_map.t())
        if not keep_grad:
            vec = self.proc_torch(vec)
        return vec

    def proc_torch(self, fn_torch):
        fn_torch = self._cuda(fn_torch)
        fn_torch = self.make_requires_grad(fn_torch)
        return fn_torch

    def make_requires_grad(self, fn_torch):
        if not fn_torch.requires_grad:
            fn_torch = Variable(fn_torch.data, requires_grad=True)
        return fn_torch

    def _cuda(self, fn_torch):
        if self.cuda and not fn_torch.is_cuda and fn_torch.requires_grad:
            return self.make_requires_grad(fn_torch.cuda())
        elif self.cuda and not fn_torch.is_cuda:
            return fn_torch.cuda()
        else:
            return fn_torch

    def is_in_spaces(self, fn_or_vec):
        return (
            self.is_V(fn_or_vec)
            or self.is_torch(fn_or_vec)
            or self.is_numpy(fn_or_vec)
            or self.is_ring(fn_or_vec)
        )


def interleave(a, b):
    c = torch.empty(len(a) + len(b)).to(a.device)
    c[0::2] = a
    c[1::2] = b
    return c


def de_interleave(c):
    a = c[0::2]
    b = c[1::2]
    return a, b
