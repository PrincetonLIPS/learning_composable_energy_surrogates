import collections
import torch


"""
def torch_namedtuple(NamedtupleClass):
    class TorchNamedTuple(NamedtupleClass):
        def __init__(self, *args, **kwargs):
            super(TorchNamedTuple, self).__init__(*args, **kwargs)
            for k, v in self._asdict().items():
                assert isinstance(v, torch.Tensor)

        def cuda(self):
            return TorchNamedTuple(**{k: v.cuda() for k, v in self._asdict()})

        def to(self, device):
            return TorchNamedTuple(**{k: v.to(device) for k, v in self._asdict()})

    TorchNamedTuple.__name__ = NamedtupleClass.__name__
    return TorchNamedTuple
"""

Example = collections.namedtuple("Example", "u p f J H guess")
Example.__new__.__defaults__ = (None,)*len(Example._fields)

"""
if __name__ == '__main__':
    example_namedtuple = collections.namedtuple("Example", "u p f J")
    print(example_namedtuple)
    print(type(example_namedtuple))
    Example = torch_namedtuple(example_namedtuple)
    print(Example)
"""
