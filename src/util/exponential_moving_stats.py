import numpy as np
import pdb


class EMSVector(object):
    def __init__(self, alpha, stat_names):
        self.stat_names = [str(sn) for sn in stat_names]
        self.ems = {sn: ExponentialMovingStats(alpha) for sn in self.stat_names}

    def feed(self, dpoints):
        assert len(dpoints) == len(self.stat_names)
        for sn, dpoint in zip(self.stat_names, dpoints):
            self.ems[sn].feed(dpoint)


class ExponentialMovingStats(object):
    def __init__(self, alpha):
        self.n = 0.0
        self.alpha = alpha
        self._mean = 0.0
        self._std2 = 0.0
        self.m90 = np.nan
        self.m50 = np.nan
        self.m10 = np.nan

    def feed(self, dpoint):
        self._mean = self.alpha * self._mean + (1.0 - self.alpha) * dpoint
        self.n += 1
        self._std2 = (
            self.alpha * self._std2 + (1.0 - self.alpha) * (self.mean - dpoint)**2
        )

        self.m90 = self.update_percentile(self.m90, dpoint, 0.9)
        self.m50 = self.update_percentile(self.m50, dpoint, 0.5)
        self.m10 = self.update_percentile(self.m10, dpoint, 0.1)

    @property
    def mean(self):
        if self.n <= 0:
            return np.nan
        return self._mean / (1 - self.alpha ** self.n)

    @property
    def std(self):
        if self.n <= 0:
            return np.nan
        return np.sqrt(self._std2 / (1 - self.alpha ** self.n))

    @property
    def delta(self):
        return self.std * 0.1

    def update_percentile(self, current, new, p):
        if not np.isfinite(current) or not np.isfinite(self.std):
            return new
        if new < current:
            return current - self.delta / p
        elif new > current:
            return current + self.delta / (1.0 - p)
        else:
            return current


if __name__ == '__main__':
    ems = ExponentialMovingStats(0.9)
    pdb.set_trace()
