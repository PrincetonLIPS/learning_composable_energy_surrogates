"""Import fenics and fenics_adjoint to one place, configure for our purposes"""
from fenics import *
from fenics_adjoint import *

set_log_level(30)

from .primal_dual_map import dCoeff_to_dField, dField_to_dCoeff

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
    if getattr(fn, "is_fa_gradient", False):
        V0 = fn.function_space()

        # Handle the special cases where only one space is BoundaryMesh
        # In short, we need to separately
        # (i) losselessly transform between boundary/line to domain/area
        # (ii) use assembly identity to transform between different meshes
        # sometimes in the opposite order

        if isinstance(fn_space.mesh(), backend.BoundaryMesh) and not isinstance(
            V0.mesh(), backend.BoundaryMesh
        ):
            # fn is over area, fn_space is over boundary i.e line
            # first make fn live on BoundaryMesh (ie line)
            # using a boundary_fn of fn.function_space() to avoid any loss
            # then use interpolate with assembly identity
            V = make_boundary_function_space(V0)
            fn.set_allow_extrapolation(True)
            fn = _old_interpolate(fn, V)
            fn.is_fa_gradient = True
            fn.set_allow_extrapolation(True)
            return interpolate(fn, fn_space)

        elif isinstance(V0.mesh(), backend.BoundaryMesh) and not isinstance(
            fn_space.mesh(), backend.BoundaryMesh
        ):
            # fn is over line / boundary, fn_space is over area
            # first create V which is a boundaryfnspace of fn_space
            # then interpolate fn to V using assembly identity
            # then _old_interpolate a boundaryexpression of this to fn_space
            # (this last step avoids info los bc V is a boundaryfnspace of fn_space)
            V = make_boundary_function_space(fn_space)
            fn = interpolate(fn, V)
            boundary_expression = make_boundary_expression(fn)
            result = _old_interpolate(boundary_expression, fn_space)
            result.is_fa_gradient(True)
            return result

        # If both spaces have same geometric dimension, can use the assembly identity
        field = dCoeff_to_dField(fn)
        field.set_allow_extrapolation(True)
        new_field = _old_interpolate(field, fn_space, *args, **kwargs)
        result = dField_to_dCoeff(new_field)
        result.is_fa_gradient = True
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
