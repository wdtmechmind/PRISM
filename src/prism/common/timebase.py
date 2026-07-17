import threading
import time
from collections import deque


class FpsMeter(object):
    def __init__(self, window=60):
        self.times = deque(maxlen=int(window))
        self.lock = threading.Lock()

    def tick(self, t=None):
        if t is None:
            t = time.time()
        with self.lock:
            self.times.append(t)

    def fps(self):
        with self.lock:
            if len(self.times) < 2:
                return 0.0
            span = self.times[-1] - self.times[0]
            if span <= 1e-9:
                return 0.0
            return (len(self.times) - 1) / span


def pick_nearest(buf, t_ref):
    best = None
    best_d = 1e18
    for ts, fr in buf:
        d = abs(ts - t_ref)
        if d < best_d:
            best_d = d
            best = (ts, fr)
    return best


def pick_bracket(buf, t_ref):
    before = None
    after = None
    for ts, fr in buf:
        if ts <= t_ref and (before is None or ts > before[0]):
            before = (ts, fr)
        if ts >= t_ref and (after is None or ts < after[0]):
            after = (ts, fr)
    return before, after
