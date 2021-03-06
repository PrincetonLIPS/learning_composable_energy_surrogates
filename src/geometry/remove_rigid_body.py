import torch
from torch import nn
from torch.nn import functional as torchF
import numpy as np
import pdb
import ast


class RigidRemover(object):
    def __init__(self, fsm):
        self.fsm = fsm
        self.ref_cpu = remove_translation(
            fsm.ring_coords.detach().clone().view(1, -1, 2)
        ).view(-1, 2)
        if fsm.cuda:
            self.ref_cuda = self.ref_cpu.cuda()

    @property
    def ref(self):
        return self.ref_cuda if self.fsm.cuda else self.ref_cpu

    def __call__(self, inputs):
        intermediates = self.fsm.to_ring(inputs, keep_grad=True)
        if len(intermediates.size()) == 2:
            intermediates = intermediates.unsqueeze(0)
            to_squeeze = True
        else:
            to_squeeze = False
        ret = remove_rotation(
            remove_translation(intermediates) + self.ref.view(1, -1, 2), self.ref,
        ) - self.ref.view(1, -1, 2)
        # pdb.set_trace()
        if to_squeeze:
            ret = ret.squeeze(0)
        if self.fsm.is_ring(inputs):
            return ret
        elif self.fsm.is_torch(inputs):
            return self.fsm.ring_to_torch(ret, keep_grad=True)
        elif self.fsm.is_numpy(inputs):
            return self.fsm.to_numpy(ret)
        else:
            raise Exception("Don't support removing rigid from other types")


def remove_translation(x):
    """Removes translations by subtracting the mean of x

    Expect x to have shape [batch_size, n_locs, 2].
    """
    assert len(x.size()) == 3 and x.size(2) == 2
    return x - torch.mean(x, dim=1, keepdim=True)


def batched_2x2_svd(A):
    assert len(A.size()) == 3 and A.size(1) == 2 and A.size(2) == 2
    U, S, V = torch.svd(A)
    return U, S, V


def batched_2x2_det(A):
    assert len(A.size()) == 3 and A.size(1) == 2 and A.size(2) == 2
    return A[:, 0, 0] * A[:, 1, 1] - A[:, 1, 0] * A[:, 0, 1]


def remove_rotation(x, ref, return_theta_R=False):
    """Finding the optimal rotation to move x close to ref
    in 2d is known as Procrustes analysis. In 3d it is known as Wahba's
    problem. See Wikipedia for a treatment.
    Variable names follow their nomenclature.

    This only requires a 2x2 SVD"""
    assert len(x.size()) == 3 and x.size(2) == 2
    assert len(ref.size()) == 2 and ref.size(1) == 2
    assert ref.size(0) == x.size(1)
    batch_size = x.size(0)
    n = x.size(1)
    ref = ref.view(1, -1, 2)

    xr_ = ref[:, :, 0]
    yr_ = ref[:, :, 1]

    w_ = x[:, :, 0]
    z_ = x[:, :, 1]

    theta = torch.atan2(
        torch.sum(w_ * yr_ - z_ * xr_, dim=1), torch.sum(w_ * xr_ + z_ * yr_, dim=1)
    )
    # print(theta)
    R = torch.stack(
        [
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1),
            torch.stack([torch.sin(theta), torch.cos(theta)], dim=1),
        ],
        dim=1,
    )

    # print(R.shape)
    # print(R)
    # B = torch.matmul(x.permute(0, 2, 1), ref)

    # U, S, V = batched_2x2_svd(B)

    # R = torch.matmul(U, V.transpose(-2, -1))

    ret = torch.matmul(R, x.permute(0, 2, 1)).permute(0, 2, 1)
    if return_theta_R:
        return ret, theta, R
    else:
        return ret
    # return torch.matmul(R.transpose(-2, -1), x.permute(0, 2, 1)).permute(0, 2, 1)


if __name__ == "__main__":
    ref = torch.Tensor(np.random.randn(16, 2))
    ref = remove_translation(ref.view(1, -1, 2)).view(-1, 2)
    thetas = np.random.randn(512)
    print(thetas)
    R = torch.Tensor(
        np.array([[np.cos(thetas), -np.sin(thetas)], [np.sin(thetas), np.cos(thetas)]])
    ).permute(2, 0, 1)
    print(R.shape)
    print(R)
    trans = torch.Tensor(np.random.randn(256, 1, 2))
    x = torch.matmul(R, ref.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)  # + trans

    x_ = x  # remove_translation(x)
    ref_ = ref  # remove_translation(ref.view(1, -1, 2)).view(-1, 2)
    x_, thetahat, Rhat = remove_rotation(x_, ref_, return_theta_R=True)
    print((x - ref.view(1, -1, 2)).norm())
    print((x_ - ref_.view(1, -1, 2)).norm())
    pdb.set_trace()
