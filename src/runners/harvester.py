import ray


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

    def step(self, init_args=(), step_args=()):
        self.sow(init_args, step_args)
        self.reap(step_args)

    def sow(self, init_args, step_args):
        while len(self.ids_to_workers) < self.max_workers:
            new_worker = self.WorkerClass.remote(self.args, *init_args)
            self.ids_to_workers[
                new_worker.step.remote(*step_args)
            ] = new_worker

    def reap(self, step_args):
        ready_ids, remaining_ids = ray.wait(
            [id for id in self.ids_to_workers.keys()], timeout=0.01
        )
        results = {id: carefully_get(id) for id in ready_ids}
        valid_results = []
        # Restart or kill workers as necessary
        for id, result in results.items():
            worker = self.ids_to_workers.pop(id)
            if not isinstance(result, Exception):
                self.ids_to_workers[
                    worker.step.remote(*step_args)
                ] = worker
                valid_results.append(result)
                self.n_success += 1
            else:
                self.n_death += 1
                self.last_error = result

        for res in valid_results:
            self.accumulator.feed(res)


def carefully_get(x):
    try:
        return ray.get(x)
    except Exception as e:
        return e
