
class EMSVector(object):
    def __init__(self, alpha, stat_names):
        self.stat_names = [str(sn) for sn in stat_names]
        self.ems = {sn: ExponentialMovingStats(alpha)
                    for sn in self.stat_names}

    def feed(self, dpoints):
        assert len(dpoints) == len(self.stat_names)
        for sn, dpoint in zip(self.stat_names, dpoints):
            self.ems[sn].feed(dpoint)


class ExponentialMovingStats(object):
    def __init__(self, alpha):
        self.n = 0.
        self.alpha = 0.
        self._mean = 0.
        self._std = 0.
        self.m90 = None
        self.m50 = None
        self.m10 = None

    def feed(self, dpoint):
        self._mean = self.alpha * self._mean + (1. - self.alpha) * dpoint
        self._std = self.alpha * self._std + (1. - self.alpha) * (self.mean - dpoint)**2

        self.m90 = self.update_percentile(self.m90, dpoint, 0.9)
        self.m50 = self.update_percentile(self.m50, dpoint, 0.5)
        self.m10 = self.update_percentile(self.m10, dpoint, 0.1)
        self.n += 1

    @property
    def mean(self):
        assert self.n > 0
        return self.mean / (1 - self.alpha**self.n)

    @property
    def std(self):
        assert self.n > 0
        return self.std / (1 - self.alpha**self.n)

    def update_percentile(self, current, new, p):
        if current is None:
            return new
        if new < current:
            return current - self.delta() / p

        elif new > current:
            return current + self.delta() / (1. - p)
        else:
            return current
