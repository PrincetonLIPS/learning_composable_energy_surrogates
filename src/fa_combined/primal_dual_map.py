from fenics_adjoint import backend, Function
import pdb


def dCoeff_to_dField(fn):
    assert isinstance(fn, Function)
    ret = Function(fn.function_space())
    u = backend.TrialFunction(fn.function_space())
    v = backend.TestFunction(fn.function_space())
    M = backend.assemble(backend.inner(u, v) * backend.dx)
    backend.solve(M, ret.vector(), fn.vector())
    return ret


def dField_to_dCoeff(fn):
    assert isinstance(fn, Function)
    ret = Function(fn.function_space())
    u = backend.TrialFunction(fn.function_space())
    v = backend.TestFunction(fn.function_space())
    M = backend.assemble(backend.inner(u, v) * backend.dx)
    ret.vector().set_local(M * fn.vector())
    return ret
