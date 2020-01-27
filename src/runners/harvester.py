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
        self.ids_to_seeds = {}
        self.seeds = set([])
        self.last_error = None
        self.last_error_time = 0

    def step(self, init_args=(), step_args=()):
        self.sow(init_args, step_args)
        self.reap(step_args)
        self.sow(init_args, step_args)

    def sow(self, init_args, step_args):
        for _ in range(self.max_workers - len(self.ids_to_workers)):
            seed = 0
            while i in self.seeds:
                seed += 1
            new_worker = self.WorkerClass.remote(
                self.args, seed, *init_args
            )
            worker_id = new_worker.step.remote(*step_args)
            self.ids_to_workers[worker_id] = new_worker
            self.seeds.add(seed)
            self.ids_to_seeds[worker_id] = seed

    def reap(self, step_args):
        ready_ids, remaining_ids = ray.wait(
            [id for id in self.ids_to_workers.keys()], timeout=1e-9
        )
        results = {id: carefully_get(id) for id in ready_ids}
        valid_results = []
        # Restart or kill workers as necessary
        for id, result in results.items():
            worker = self.ids_to_workers.pop(id)
            seed = self.ids_to_seeds.pop(id)
            if not isinstance(result, Exception):
                worker_id = worker.step.remote(*step_args)
                self.ids_to_workers[worker_id] = worker
                self.ids_to_seeds[worker_id] = seed
                valid_results.append(result)
                self.n_success += 1
            else:
                self.n_death += 1
                self.last_error = result
                self.last_error_time = time.time()
                self.seeds.remove(seed)

        for res in valid_results:
            self.accumulator(res)


def carefully_get(x):
    try:
        return ray.get(x)
    except Exception as e:
        return e
