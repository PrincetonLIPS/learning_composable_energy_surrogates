import ray
import time
import numpy as np

MAX_SEED = 2 ** 32


class Harvester(object):
    def __init__(self, args, accumulator, WorkerClass, max_workers):
        self.args = args
        self.accumulator = accumulator
        self.WorkerClass = WorkerClass
        self.max_workers = max_workers
        self.n_success = 0
        self.n_death = 0
        self.ids_to_workers = {}
        self.last_error = None
        self.last_error_time = 0
        self.zero_id = None

    def step(self, init_args=(), step_args=()):
        self.sow(init_args, step_args)
        self.reap(step_args)
        self.sow(init_args, step_args)

    def sow(self, init_args, step_args):
        for _ in range(self.max_workers - len(self.ids_to_workers)):
            if self.zero_id is None:
                seed = 0
                new_worker = self.WorkerClass.remote(
                    self.args, np.random.randint(2**32), *init_args
                )
                id = new_worker.step.remote(*step_args)
                self.zero_id = id
                self.ids_to_workers[id] = new_worker
            else:
                seed = np.random.randint(2**32)
                new_worker = self.WorkerClass.remote(
                    self.args, seed, *init_args
                )
                self.ids_to_workers[new_worker.step.remote(*step_args)] = new_worker
            print("Started {} with seed {}".format(super(self.WorkerClass.__bases__, seed))

    def reap(self, step_args):
        ready_ids, remaining_ids = ray.wait(
            [id for id in self.ids_to_workers.keys()], timeout=1e-9
        )
        results = {id: carefully_get(id) for id in ready_ids}
        valid_results = []
        # Restart or kill workers as necessary
        for id, result in results.items():
            worker = self.ids_to_workers.pop(id)
            if not isinstance(result, Exception):
                self.ids_to_workers[worker.step.remote(*step_args)] = worker
                valid_results.append(result)
                self.n_success += 1
            else:
                self.n_death += 1
                self.last_error = result
                self.last_error_time = time.time()
                if id == self.zero_id:
                    self.zero_id = None

        for res in valid_results:
            self.accumulator(res)


def carefully_get(x):
    try:
        return ray.get(x)
    except Exception as e:
        return e
