# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import time
from collections import OrderedDict
from math import log


class DummyStatsContext:
    """
    Dummy statistics context, doing nothing.
    """
    def __init__(self, stats, what):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


class DummyStats:
    """
    Dummy statistics, doing nothing.
    """
    def __init__(self, bkz, verbose=False):
        pass

    def context(self, what, **kwds):
        return DummyStatsContext(self, what)


class BKZStatsContext:
    def __init__(self, stats, what, **kwds):
        self.stats = stats
        self.what = what
        self.kwds = kwds

    def __enter__(self):
        self.timestamp = time.clock()
        self.stats.begin(self.what, **self.kwds)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        time_spent = time.clock() - self.timestamp
        self.stats.end(self.what, time_spent, **self.kwds)


class BKZStats:
    def __init__(self, bkz, verbose=False):
        self.bkz = bkz
        self.verbose = verbose
        self.i = 0
        self.tours = []
        self.mutable = True

    def _check_mutable(self):
        if not self.mutable:
            raise ValueError("This stats object is immutable.")

    def context(self, what, **kwds):
        self._check_mutable()
        return BKZStatsContext(self, what, **kwds)

    def _tour_begin(self):
        i = self.i
        self.tours.append(OrderedDict([("i", i),
                                       ("total time", 0.0),
                                       ("time", 0.0),
                                       ("preproc time", 0.0),
                                       ("svp time", 0.0),
                                       ("lll time", 0.0),
                                       ("postproc time", 0.0),
                                       ("r_0", 0.0),
                                       ("slope", 0.0),
                                       ("enum nodes", 1),
                                       ("max(kappa)", 0)]))

        if i > 0:
            self.tours[i]["total time"] = self.tours[i-1]["total time"]
            self.tours[i]["max(kappa)"] = self.tours[i-1]["max(kappa)"]

    def _tour_end(self, time):
        i = self.i
        self.tours[i]["time"] = time
        r, exp = self.bkz.M.get_r_exp(0, 0)
        self.tours[i]["r_0"] = r * 2 **exp
        self.tours[i]["slope"] = self.bkz.M.get_current_slope(0, self.bkz.A.nrows)
        self.tours[i]["total time"] += self.tours[i]["time"]

        if self.verbose:
            print(self)
        self.i += 1

    def _svp_end(self, time, E=None):
        self.tours[self.i]["svp time"] += time
        if E:
            self.tours[self.i]["enum nodes"] += E.get_nodes()

    def begin(self, what, **kwds):
        self._check_mutable()

        if what == "tour":
            return self._tour_begin()

        if self.i > len(self.tours):
            self._tour_begin()
        if "%s time"%what not in self.tours[self.i]:
            self.tours[self.i]["%s time"%what] = 0.0

    def end(self, what, time, *args, **kwds):
        self._check_mutable()

        if what == "tour":
            return self._tour_end(time, *args, **kwds)
        elif what == "svp":
            return self._svp_end(time, *args, **kwds)

        self.tours[self.i]["%s time"%what] += time

    @property
    def current_tour(self):
        return self.tours[self.i]

    def log_clean_kappa(self, kappa, clean):
        self._check_mutable()
        if clean and self.tours[self.i]["max(kappa)"] < kappa:
            self.tours[self.i]["max(kappa)"] = kappa

    def finalize(self):
        """
        Data collection is finished.
        """
        self.mutable = False

    def dumps_tour(self, i):
        tour = self.tours[i]
        s = []
        s.append("\"i\": %3d"%i)
        s.append("\"total\": %9.2f"%(tour["total time"]))
        s.append("\"time\": %8.2f"%(tour["time"]))
        s.append("\"preproc\": %8.2f"%(tour["preproc time"]))
        s.append("\"svp\": %8.2f"%(tour["svp time"]))
        s.append("\"lll\": %8.2f"%(tour["lll time"]))
        s.append("\"postproc\": %8.2f"%(tour["postproc time"]))
        s.append("\"r_0\": %.4e"%(tour["r_0"]))
        s.append("\"slope\": %7.4f"%(tour["slope"]))
        s.append("\"enum nodes\": %5.2f"%(log(tour["enum nodes"], 2)))
        s.append("\"max(kappa)\": %3d"%(tour["max(kappa)"]))
        return "{" + ",  ".join(s) + "}"

    def __str__(self):
        return self.dumps_tour(len(self.tours)-1)

    @property
    def enum_nodes(self):
        """
        Total number of nodes visited during enumeration.
        """
        nodes = 0
        for tour in self.tours:
            nodes += tour["enum nodes"]
        return nodes

    @property
    def total_time(self):
        """
        Total time spent.
        """
        return self.tours[-1]["total time"]

    @property
    def svp_time(self):
        """
        Total time spent in SVP oracle.
        """
        time = 0
        for tour in self.tours:
            time += tour["svp time"]
        return time

    @property
    def lll_time(self):
        """
        Total time spent in LLL.
        """
        time = 0
        for tour in self.tours:
            time += tour["lll time"]
        return time
