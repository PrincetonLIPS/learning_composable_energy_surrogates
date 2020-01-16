from .. import fa_combined as fa


# Infintesimal strain
def InfintesimalStrain(u):
    return fa.variable(0.5 * (fa.grad(u) + fa.grad(u).T))


# Deformation gradient
def DeformationGradient(u):
    I = fa.Identity(u.geometric_dimension())
    return fa.variable(I + fa.grad(u))


# Determinant of the deformation gradient
def DetDeformationGradient(u):
    F = DeformationGradient(u)
    return fa.variable(fa.det(F))


# Right Cauchy-Green tensor
def RightCauchyGreen(u):
    F = DeformationGradient(u)
    return fa.variable(F.T * F)


# Green-Lagrange strain tensor
def GreenLagrangeStrain(u):
    I = fa.Identity(u.geometric_dimension())
    C = RightCauchyGreen(u)
    return fa.variable(0.5 * (C - I))


# Invariants of an arbitrary tensor, A
def Invariants(A):
    I1 = fa.tr(A)
    I2 = 0.5 * (fa.tr(A) ** 2 - fa.tr(A * A))
    I3 = fa.det(A)
    return [I1, I2, I3]


# Isochoric part of the deformation gradient
def IsochoricDeformationGradient(u):
    F = DeformationGradient(u)
    J = DetDeformationGradient(u)
    return fa.variable(J ** (-1.0 / 3.0) * F)


# Isochoric part of the right Cauchy-Green tensor
def IsochoricRightCauchyGreen(u):
    C = RightCauchyGreen(u)
    J = DetDeformationGradient(u)
    return fa.variable(J ** (-2.0 / 3.0) * C)


def Linear_Energy(u, lam, mu, return_stress=False):
    strain = InfintesimalStrain(u)
    energy = (lam / 2.0) * (fa.tr(strain) ** 2) + mu * fa.inner(strain, strain)
    if return_stress:
        I = fa.Identity(u.geometric_dimension())
        cauchy_stress = lam * fa.tr(strain) * I + 2 * mu * strain
        return energy, cauchy_stress
    return energy


def SVK_Energy(u, lam, mu, return_stress=False):
    E = GreenLagrangeStrain(u)
    energy = (lam / 2.0) * (fa.tr(E) ** 2) + mu * fa.inner(E, E)
    if return_stress:
        second_pk_stress = fa.diff(energy, E)
        return energy, second_pk_stress
    return energy


def MooneyRiven_Energy(u, c1, c2, return_stress=False):
    C = RightCauchyGreen(u)
    I1, I2, I3 = Invariants(C)
    energy = c1 * (I1 - 3) + c2 * (I2 - 3)
    if return_stress:
        I = fa.Identity(u.geometric_dimension())
        gamma1 = fa.diff(energy, I1) + I1 * fa.diff(energy, I2)
        gamma2 = -fa.diff(energy, I2)
        gamma3 = I3 * fa.diff(energy, I3)
        second_pk_stress = 2 * (gamma1 * I + gamma2 * C + gamma3 * fa.inv(C))
        return energy, second_pk_stress
    return energy


"""
How to do NHE from DoFs:
- map from DoFs to piecewise constant gradient within squares
- bsize * n_squares * dim
- compute DefGrad F for squares
- bsize * n_squares * dim * dim
- compute invariants, I1 = tr(F.T * F), J = det(F)
- bsize * n_squares * 1
- compute energy density within squares
- bsize * n_squares * 1
- compute energy = sum_squares ed * square_area
"""


def torchDet(x):
    assert len(x.size()) >= 2
    assert x.size(-1) == x.size(-2)
    if x.size(-1) == 1:
        return x.squeeze(-1)
    elif x.size(-1) == 2:
        return x[:, :, 0, 0] * x[:, :, 1, 1] - x[:, :, 0, 1] * x[:, :, 1, 0]
    else:
        raise Exception("Haven't implemented det for dim {}".format(x.size(-1)))


def PTNeoHookeanEnergy(grad_u, young_mod, poisson_ratio, elem_weights=None):
    """Pytorch NeoHookeanEnergy

    Inputs:
        grad_u: Tensor [batch_sze, n_elems, dim, dim]
        young_mod: Float
        poisson_ratio: Float
        elem_weights: Tensor [batch_size or 1, n_elems]. Assumed uniform if None
    """
    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
                poisson_ratio
            )
        )

    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))
    d = grad_u.size(2)

    # Deformation Gradient
    I = torch.eye(d).view(1, 1, d, d).repeat(grad_u.size(0), grad_u.size(1), 1, 1)

    F = I + grad_u

    J = torchDet(F)

    Jinv = J ** (-2 / d)

    right_cauchy_green = torch.matmul(torch.transpose(F, -1, -2), F)

    I1 = (I * right_cauchy_green).sum(dim=3).sum(dim=2, keepdims=True)

    energy = (shear_mod / 2) * (Jinv * I1 - d) + (bulk_mod / 2) * (J - 1) ** 2
    return energy


def NeoHookeanEnergy(u, young_mod, poisson_ratio, return_stress=False):
    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
                poisson_ratio
            )
        )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))
    d = u.geometric_dimension()
    F = DeformationGradient(u)
    J = fa.det(F)
    Jinv = J ** (-2 / d)
    I1 = fa.tr(RightCauchyGreen(u))
    energy = (shear_mod / 2) * (Jinv * I1 - d) + (bulk_mod / 2) * (J - 1) ** 2
    if return_stress:
        FinvT = fa.inv(F).T
        first_pk_stress = (
            Jinv * shear_mod * (F - (1 / d) * I1 * FinvT)
            + J * bulk_mod * (J - 1) * FinvT
        )
        return energy, first_pk_stress

    return energy
