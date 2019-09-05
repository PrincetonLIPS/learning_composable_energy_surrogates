import collections
import torch


def torch_namedtuple(namedtuple_class):
    class TorchNamedTuple(namedtuple_class):
        def __init__(self, *args, **kwargs):
            super(TorchNamedTuple, self).__init__(*args, **kwargs)
            for k, v in self._asdict().items():
                assert isinstance(v, torch.Tensor)

        def cuda(self):
            return TorchNamedTuple(**{k: v.cuda() for k, v in self._asdict()})

        def to(self, device):
            return TorchNamedTuple(**{k: v.to(device) for k, v in self._asdict()})


Example = torch_namedtuple(collections.namedtuple("Example", "u p f J"))
