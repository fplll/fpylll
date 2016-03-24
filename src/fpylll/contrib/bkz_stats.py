# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import time
from collections import OrderedDict


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
        self.tours = {}

    def context(self, what):
        return BKZStatsContext(self, what)

    def tour_begin(self):
        i = self.i
        self.tours[i] = OrderedDict([("i", i),
                                     ("total time", 0.0),
                                     ("time", 0.0),
                                     ("preproc time", 0.0),
                                     ("svp time", 0.0),
                                     ("postproc time", 0.0),
                                     ("r_0", 0.0),
                                     ("slope", 0.0),
                                     ("max(kappa)", 0)])

        if i > 0:
            self.tours[i]["total time"] = self.tours[i-1]["total time"]
            self.tours[i]["max(kappa)"] = self.tours[i-1]["max(kappa)"]

    def tour_end(self, time):
        i = self.i
        self.tours[i]["time"] = time
        r, exp = self.bkz.m.get_r_exp(0, 0)
        self.tours[i]["r_0"] = r * 2 **exp
        self.tours[i]["slope"] = self.bkz.m.get_current_slope(0, self.bkz.A.nrows)
        self.tours[i]["total time"] += self.tours[i]["time"]

        if self.verbose:
            print(self)
        self.i += 1

    def preproc_begin(self):
        if self.i not in self.tours:
            self.tour_begin()

    def preproc_end(self, time):
        self.tours[self.i]["preproc time"] += time

    def svp_begin(self):
        if self.i not in self.tours:
            self.tour_begin()

    def svp_end(self, time):
        self.tours[self.i]["svp time"] += time

    def postproc_begin(self):
        if self.i not in self.tours:
            self.tour_begin()

    def postproc_end(self, time):
        i = self.i
        self.tours[i]["postproc time"] += time

    def __str__(self):
        s = []
        tour = self.tours[self.i]
        s.append("\"i\": %3d"%self.i)
        s.append("\"total\": %9.2f"%(tour["total time"]))
        s.append("\"time\": %8.2f"%(tour["time"]))
        s.append("\"preproc\": %8.2f"%(tour["preproc time"]))
        s.append("\"svp\": %8.2f"%(tour["svp time"]))
        s.append("\"r_0\": %.4e"%(tour["r_0"]))
        s.append("\"slope\": %7.4f"%(tour["slope"]))
        s.append("\"max(kappa)\": %3d"%(tour["max(kappa)"]))
        return "{" + ",  ".join(s) + "}"

    def log_clean_kappa(self, kappa, clean):
        if clean and self.tours[self.i]["max(kappa)"] < kappa:
            self.tours[self.i]["max(kappa)"] = kappa
