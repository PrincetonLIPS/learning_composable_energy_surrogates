import numpy as np
from torch.util.data import Dataset
from example import Example


class DataBuffer(Dataset):
    def __init__(self, memory_size, batch_size,
                 drop_prob=0, to_np=True, safe_idx=0):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.safe_idx = safe_idx

    def feed(self, example):
        assert isinstance(example, Example)
        if np.random.rand() < self.drop_prob:
            return
        if (self.pos + self.safe_idx) >= len(self.data):
            self.data.append((example.u, example.p, example.f, example.J))
        else:
            self.data[self.pos + self.safe_idx] = (
                example
            )
        self.pos = (self.pos + 1) % (self.memory_size - self.safe_idx)

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
