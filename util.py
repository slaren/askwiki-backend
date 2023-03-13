import time


class perf_logger:
    def __init__(self, name, logger):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.tstart = time.monotonic()
        return self

    def __exit__(self, *exc):
        ttaken = time.monotonic() - self.tstart
        self.logger.info("%s: %.3f seconds", self.name, ttaken)
