# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import time
from collections import OrderedDict
from math import log
from fpylll.enumeration import Enumeration


class BKZStatsContext:
    def __init__(self, stats, what):
        self.stats = stats
        self.what = what

    def __enter__(self):
        self.timestamp = time.clock()

        if self.what == "tour":
            self.stats.tour_begin()
        elif self.what == "preproc":
            self.stats.preproc_begin()
        elif self.what == "svp":
            self.stats.svp_begin()
        elif self.what == "postproc":
            self.stats.postproc_begin()
        else:
            raise NotImplementedError

    def __exit__(self, exception_type, exception_value, exception_traceback):
        time_spent = time.clock() - self.timestamp
        if self.what == "tour":
            self.stats.tour_end(time_spent)
        elif self.what == "preproc":
            self.stats.preproc_end(time_spent)
        elif self.what == "svp":
            self.stats.svp_end(time_spent)
        elif self.what == "postproc":
            self.stats.postproc_end(time_spent)
        else:
            raise NotImplementedError


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

    def context(self, what):
        self._check_mutable()
        return BKZStatsContext(self, what)

    def tour_begin(self):
        self._check_mutable()
        i = self.i
        self.tours.append(OrderedDict([("i", i),
                                       ("total time", 0.0),
                                       ("time", 0.0),
                                       ("preproc time", 0.0),
                                       ("svp time", 0.0),
                                       ("postproc time", 0.0),
                                       ("r_0", 0.0),
                                       ("slope", 0.0),
                                       ("enum nodes", 0),
                                       ("max(kappa)", 0)]))

        if i > 0:
            self.tours[i]["total time"] = self.tours[i-1]["total time"]
            self.tours[i]["max(kappa)"] = self.tours[i-1]["max(kappa)"]

    def tour_end(self, time):
        self._check_mutable()
        i = self.i
        self.tours[i]["time"] = time
        r, exp = self.bkz.M.get_r_exp(0, 0)
        self.tours[i]["r_0"] = r * 2 **exp
        self.tours[i]["slope"] = self.bkz.M.get_current_slope(0, self.bkz.A.nrows)
        self.tours[i]["total time"] += self.tours[i]["time"]

        if self.verbose:
            print(self)
        self.i += 1

    def preproc_begin(self):
        self._check_mutable()
        if self.i > len(self.tours):
            self.tour_begin()

    def preproc_end(self, time):
        self._check_mutable()
        self.tours[self.i]["preproc time"] += time

    def svp_begin(self):
        self._check_mutable()
        if self.i > len(self.tours):
            self.tour_begin()

    def svp_end(self, time):
        self._check_mutable()
        self.tours[self.i]["svp time"] += time
        self.tours[self.i]["enum nodes"] += Enumeration.get_nodes()

    def postproc_begin(self):
        self._check_mutable()
        if self.i > len(self.tours):
            self.tour_begin()

    def postproc_end(self, time):
        self._check_mutable()
        i = self.i
        self.tours[i]["postproc time"] += time

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
