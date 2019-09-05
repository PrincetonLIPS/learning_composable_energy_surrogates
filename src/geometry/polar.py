import torch
import numpy as np
import pdb
from ..maps.function_space_map import FunctionSpaceMap
from .. import fa_combined as fa


class PolarBase(object):
    def __init__(self, fsm):
        self.fsm = fsm
        self.ref_cpu = fsm.ring_coords.detach().clone() - torch.Tensor([[0.5, 0.5]])
        self.normal_cpu = torch.stack([self.ref_cpu[:, 1], -self.ref_cpu[:, 0]], dim=1)
        if fsm.cuda:
            self.ref_cuda = self.ref_cpu.cuda()
            self.normal_cuda = self.normal_cpu.cuda()

    @property
    def ref(self):
        return self.ref_cuda if self.fsm.cuda else self.ref_cpu

    @property
    def normal(self):
        return self.normal_cuda if self.fsm.cuda else self.normal_cpu

    def decompose(self, x):
        rad_component = torch.sum(x * self.ref.unsqueeze(0), dim=2) / (
            torch.norm(self.ref.unsqueeze(0), dim=2)
        )

        tang_component = torch.sum(x * self.normal.unsqueeze(0), dim=2) / (
            torch.norm(self.normal.unsqueeze(0), dim=2)
        )
        return rad_component, tang_component

    def postprocess(self, ret, inputs):
        if self.fsm.is_ring(inputs):
            return ret
        elif self.fsm.is_torch(inputs):
            return self.fsm.ring_to_torch(ret, keep_grad=True)
        elif self.fsm.is_numpy(inputs):
            return self.fsm.to_numpy(ret)
        else:
            raise Exception("Don't support (de)polarizing from other types")


class Polarizer(PolarBase):
    def polarize(self, inputs):
        x = self.fsm.to_ring(inputs)
        r = torch.norm(x, dim=2)
        rad_component, tang_component = self.decompose(x)
        theta = torch.atan2(tang_component, rad_component)
        ret = torch.stack([r, theta], dim=2)
        return self.postprocess(ret, inputs)

    def depolarize(self, inputs):
        x = self.fsm.to_ring(inputs)
        r = x[:, :, 0]
        theta = x[:, :, 1]
        rad = self.ref.view(1, -1, 2)
        rad = rad / rad.norm(dim=2, keepdim=True)
        tang = self.normal.view(1, -1, 2)
        tang = tang / tang.norm(dim=2, keepdim=True)

        ret = (r * torch.cos(theta)).unsqueeze(2) * rad + (
            r * torch.sin(theta)
        ).unsqueeze(2) * tang

        return self.postprocess(ret, inputs)


class SemiPolarizer(PolarBase):
    def polarize(self, inputs):
        x = self.fsm.to_ring(inputs)

        rad_component, tang_component = self.decompose(x)

        ret = torch.stack([rad_component, tang_component], dim=2)
        return self.postprocess(ret, inputs)

    def depolarize(self, inputs):
        x = self.fsm.to_ring(inputs)
        rad_size = x[:, :, 0]
        tang_size = x[:, :, 1]
        rad = self.ref.view(1, -1, 2)
        rad = rad_size.unsqueeze(2) * rad / rad.norm(dim=2, keepdim=True)
        tang = self.normal.view(1, -1, 2)
        tang = tang_size.unsqueeze(2) * tang / tang.norm(dim=2, keepdim=True)

        ret = rad + tang

        return self.postprocess(ret, inputs)


if __name__ == "__main__":
    x = torch.Tensor(np.random.randn(32, 16, 2))
    m = fa.UnitSquareMesh(10, 10)
    V = fa.VectorFunctionSpace(m, "P", 2)
    fsm = FunctionSpaceMap(V, 5, 5)

    polarizer = Polarizer(fsm)
    semipolarizer = SemiPolarizer(fsm)

    x2 = polarizer.depolarize(polarizer.polarize(x))
    x3 = semipolarizer.depolarize(semipolarizer.polarize(x))

    pdb.set_trace()
