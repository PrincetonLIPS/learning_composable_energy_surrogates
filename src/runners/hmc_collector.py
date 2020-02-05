import ray
from .hmc_collector_base import HMCCollectorBase


@ray.remote(resources={"WorkerFlags": 0.33})
class HMCCollector(HMCCollectorBase):
    pass


if __name__ == "__main__":
    print("no segfault yet")
    from ..arguments import parser
    import pdb

    args = parser.parse_args()
    collector = HMCCollectorBase(args, 0)
    print(collector.step())
    pdb.set_trace()
