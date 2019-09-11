"""Import fenics and fenics_adjoint to one place, configure for our purposes"""
from fenics import *
from fenics_adjoint import *

import logging
import dijitso

logging.getLogger("UFL").setLevel(logging.WARNING)
logging.getLogger("FFC").setLevel(logging.WARNING)

dijitso.set_log_level(40)

set_log_level(30)

# Wrap compute_gradient and compute_hessian, so that the outputs have
# an attribute which tells us they're a gradient or hessian.
_old_compute_gradient = compute_gradient
_old_compute_hessian = compute_hessian
_old_interpolate = interpolate


def make_boundary_expression(function):
    element = function.ufl_element()
    function.set_allow_extrapolation(True)
    mesh = function.function_space().mesh()
    bbt = mesh.bounding_box_tree()
    __MAX_UINT__ = 4294967295

    class BoundaryExpression(backend.UserExpression):
        def eval(self, value, x):
            id2, _ = bbt.compute_closest_entity(backend.Point(x))
            if (
                backend.near(x[0], 0.5) and backend.near(x[1], 0.5)
            ) or id2 >= __MAX_UINT__:
                value = [0.0 for _ in range(len(value))]
            else:
                v = function(x)
                if hasattr(v, "__iter__"):
                    for i in range(len(v)):
                        value[i] = v[i]
                else:
                    value[0] = v

        def value_shape(self):
            return (function.function_space().ufl_element().value_size(),)

    return BoundaryExpression(element=element)


def interpolate(fn, fn_space, *args, **kwargs):
    if getattr(fn, "is_fa_gradient", False) or getattr(fn, "is_fa_hessian", False):
        raise Exception(
            "Don't do this! "
            "pyadjoint gives gradient w.r.t. params, "
            "not gradient field. Fenics interpolate will "
            "treat this input as a field."
        )
    else:
        result = _old_interpolate(fn, fn_space, *args, **kwargs)
    return result


def make_boundary_function_space(V0):
    assert isinstance(V0, backend.FunctionSpace)
    if isinstance(V0.mesh(), backend.BoundaryMesh):
        return V0
    bmesh = backend.BoundaryMesh(V0.mesh(), "exterior")
    if V0.ufl_element().value_size() == 1:
        V = backend.FunctionSpace(
            bmesh, V0.ufl_element().family(), V0.ufl_element().degree()
        )
    else:
        V = backend.VectorFunctionSpace(
            bmesh,
            V0.ufl_element().family(),
            V0.ufl_element().degree(),
            dim=V0.ufl_element().value_size(),
        )
    return V


def compute_gradient(*args, **kwargs):
    result = _old_compute_gradient(*args, **kwargs)
    result.is_fa_gradient = True
    return result


def compute_hessian(*args, **kwargs):
    result = _old_compute_hessian(*args, **kwargs)
    result.is_fa_hvp = True
    return result
