class Dummy(object):
    '''This class does nothing in response to every message call.

    E.g.
        dummy = Dummy()
        dummy.do_something()
                ---> None

    For object oriented code, if you would otherwise call some
    class methods to change some global state (e.g. optimizer.step(),
    scheduler.step(), harvester.step().. a lot of .step()) or you
    would call some method to be used in a print statement, but you
    want to skip that computation without having to modify code in
    many places, just replace the object with a Dummy().
    '''
    def __init__(self):
        pass

    def __getattr__(self, attr):
        return lambda *args, **kwargs: [None]
