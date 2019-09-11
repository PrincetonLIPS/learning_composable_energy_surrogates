import numpy as np
from torch.utils.data import Dataset
from .example import Example


class DataBuffer(Dataset):
    def __init__(self, memory_size, safe_idx=0):
        self.memory_size = memory_size
        self.data = []
        self.pos = self.safe_idx
        self.safe_idx = safe_idx
        assert self.safe_idx >= 0
        assert self.safe_idx < self.memory_size

    def feed(self, example):
        assert isinstance(example, Example)
        if len(self.data) <= self.memory_size:
            self.data.append((example.u, example.p, example.f, example.J))
        else:
            self.data[self.pos] = example
            self.pos += 1
            if self.pos >= self.memory_size:
                self.pos = self.safe_idx

    def feed_batch(self, examples):
        for ex in examples:
            self.feed(ex)

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def __len__(self):
        return self.memory_size

    def __getitem__(self, idx):
        return self.data[idx % self.size()]
