import ray


def carefully_get(x):
    try:
        return ray.get(x)
    except Exception as e:
        return e
