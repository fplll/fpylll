# -*- coding: utf-8 -*-
"""
Collecting traces from BKZ-like computations

.. moduleauthor:: Martin R. Albrecht <fplll-devel@googlegroups.com>

"""
from __future__ import print_function
from __future__ import absolute_import

import time

try:
    from time import process_time  # Python 3
except ImportError:
    from time import clock as process_time  # Python 2
import copy
from collections import OrderedDict
from math import log
from fpylll.tools.quality import basis_quality


def pretty_dict(d, keyword_width=None, round_bound=9999, suppress_length=128):
    """Return 'pretty' string representation of the dictionary ``d``.

    :param d: a dictionary
    :param keyword_width: width allocated for keywords
    :param round_bound: values beyond this bound are shown as `2^x`
    :param suppress_length: don't print arbitrary data with ``len(str(data)) > suppress_length``

    >>> from collections import OrderedDict
    >>> str(pretty_dict(OrderedDict([('d', 2), ('f', 0.1), ('large', 4097)])))
    '{"d":        2,  "f": 0.100000,  "large":     4097}'

    """
    s = []
    for k in d:

        v = d[k]

        if keyword_width:
            fmt = u'"%%%ds"' % keyword_width
            k = fmt % k
        else:
            k = '"%s"' % k

        if isinstance(v, int):
            if abs(v) > round_bound:
                s.append(u"%s: %8s" % (k, u"%s2^%.1f" % ("" if v > 0 else "-", log(abs(v), 2))))
            else:
                s.append(u"%s: %8d" % (k, v))
            continue
        elif not isinstance(v, float):
            try:
                v = float(v)
            except TypeError:
                if len(str(v)) <= suppress_length:
                    s.append(u"%s: %s" % (k, v))
                else:
                    s.append(u"%s: '...'" % (k,))
                continue

        if 0 <= v < 10.0:
            s.append(u"%s: %8.6f" % (k, v))
        elif -10 < v < 0:
            s.append(u"%s: %8.5f" % (k, v))
        elif abs(v) < round_bound:
            s.append(u"%s: %8.3f" % (k, v))
        else:
            s.append(u"%s: %8s" % (k, u"%s2^%.1f" % ("" if v > 0 else "-", log(abs(v), 2))))

    return u"{" + u",  ".join(s) + u"}"


class Accumulator(object):
    """
    An ``Accumulator`` collects observed facts about some random variable (e.g. running time).

    In particular,

        - minimum,

        - maximum,

        - mean and

        - variance

    are recorded::

        >>> v = Accumulator(1.0); v
        1.0
        >>> v += 2.0; v
        3.0
        >>> v = Accumulator(-5.4, repr="avg"); v
        -5.4
        >>> v += 0.2
        >>> v += 5.2; v
        0.0
        >>> v.min, v.max
        (-5.4, 5.2)

    """

    def __init__(self, value, repr="sum", count=True, bessel_correction=False):
        """
        Create a new instance.

        :param value: some initial value
        :param repr: how to represent the data: "min", "max", "avg", "sum" and "variance" are
            valid choices
        :param count: if ``True`` the provided value is considered as an observed datum, i.e. the
            counter is increased by one.
        :param bessel_correction: apply Bessel's correction to the variance.
        """

        self._min = value
        self._max = value
        self._sum = value
        self._sqr = value * value
        self._ctr = 1 if count else 0
        self._repr = repr
        self._bessel_correction = bessel_correction

    def add(self, value):
        """
        Add value to the accumulator.

        >>> v = Accumulator(10.0)
        >>> v.add(5.0)
        15.0

        :param value: some value
        :returns: itself

        """
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._sum += value
        self._sqr += value * value
        self._ctr += 1
        return self

    @property
    def min(self):
        """
        >>> v = Accumulator(2.0)
        >>> v += 5.0
        >>> v.min
        2.0

        """
        return self._min

    @property
    def max(self):
        """
        >>> v = Accumulator(2.0)
        >>> v += 5.0
        >>> v.max
        5.0

        """
        return self._max

    @property
    def avg(self):
        """
        >>> v = Accumulator(2.0)
        >>> v += 5.0
        >>> v.avg
        3.5

        """
        return self._sum / self._ctr

    mean = avg

    @property
    def sum(self):
        """
        >>> v = Accumulator(2.0)
        >>> v += 5.0
        >>> v.sum
        7.0

        """
        return self._sum

    @property
    def variance(self):
        """
        >>> v = Accumulator(2.0)
        >>> v += 5.0
        >>> v.variance
        2.25

        """
        s = self._sqr / self._ctr - self.avg ** 2
        if self._bessel_correction:
            return self._ctr * (s / (self._ctr - 1))
        else:
            return s

    def __add__(self, other):
        """
        Addition semantics are:

        - ``stat + None`` returns ``stat``
        - ``stat + stat`` returns the sum of their underlying values
        - ``stat + value`` inserts ``value`` into ``stat``

        >>> v = Accumulator(2.0)
        >>> v + None
        2.0
        >>> v + v
        4.0
        >>> v + 3.0
        5.0

        """
        if other is None:
            return copy.copy(self)
        elif not isinstance(other, Accumulator):
            ret = copy.copy(self)
            return ret.add(other)
        else:
            if self._repr != other._repr:
                raise ValueError("%s != %s" % (self._repr, other._repr))
            ret = Accumulator(0)
            ret._min = min(self.min, other.min)
            ret._max = max(self.max, other.max)
            ret._sum = self._sum + other._sum
            ret._sqr = self._sqr + other._sqr
            ret._ctr = self._ctr + other._ctr
            ret._repr = self._repr
            return ret

    def __radd__(self, other):
        """
        Revert to normal addition.
        """
        return self + other

    def __sub__(self, other):
        """
        Return the difference of the two nodes reduced to floats.
        """
        return float(self) - float(other)

    def __float__(self):
        """
        Reduce this stats object down a float depending on strategy chosen in constructor.

        >>> v = Accumulator(2.0, "min"); v += 3.0; float(v)
        2.0
        >>> v = Accumulator(2.0, "max"); v += 3.0; float(v)
        3.0
        >>> v = Accumulator(2.0, "avg"); v += 3.0; float(v)
        2.5
        >>> v = Accumulator(2.0, "sum"); v += 3.0; float(v)
        5.0
        >>> v = Accumulator(2.0, "variance"); v += 3.0; float(v)
        0.25
        """
        return float(self.__getattribute__(self._repr))

    def __str__(self):
        """
        Reduce this stats object down a float depending on strategy chosen in constructor.

        >>> v = Accumulator(2.0, "min"); v += 3.0; str(v)
        '2.0'
        >>> v = Accumulator(2.0, "max"); v += 3.0; str(v)
        '3.0'
        >>> v = Accumulator(2.0, "avg"); v += 3.0; str(v)
        '2.5'
        >>> v = Accumulator(2.0, "sum"); v += 3.0; str(v)
        '5.0'
        >>> v = Accumulator(2.0, "variance"); v += 3.0; str(v)
        '0.25'
        """
        return str(self.__getattribute__(self._repr))

    __repr__ = __str__


class TraceContext(object):
    """
    A trace context collects data about an underlying process on entry/exit of particular parts of
    the code.
    """

    def __init__(self, tracer, *args, **kwds):
        """Create a new context for gathering statistics.

        :param tracer: a tracer object
        :param args: all args form a label for the trace context
        :param kwds: all kwds are considered auxiliary data

        """
        self.tracer = tracer
        self.what = args if len(args) > 1 else args[0]
        self.kwds = kwds

    def __enter__(self):
        """
        Call ``enter`` on trace object
        """
        self.tracer.enter(self.what, **self.kwds)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Call ``exit`` on trace object
        """
        self.tracer.exit(**self.kwds)

        if exception_type is not None:
            return False


class Tracer(object):
    """
    A trace object is used to collect information about processes.

    This base class does nothing.
    """

    def __init__(self, instance, verbosity=False, max_depth=16):
        """
        Create a new tracer instance.

        :param instance: BKZ-like object instance
        :param verbosity: print information, integers ≥ 0 are also accepted
        :param max_depth: record up to this depth.

        """
        self.instance = instance
        self.verbosity = int(verbosity)
        self.max_depth = max_depth

    def context(self, *args, **kwds):
        """
        Return a new ``TraceCotext``.
        """
        return TraceContext(self, *args, **kwds)

    def enter(self, label, **kwds):
        """
        An implementation would implement this function which controls what happens when the context
        given by ``label`` is entered.
        """
        pass

    def exit(self, **kwds):
        """
        An implementation would implement this function which controls what happens when the context
        given by ``label`` is left.
        """
        pass

    def _pop(self):
        # NOTE: we handle ``max_depth`` by removing children when we exit.
        child = self.current
        self.current = self.current.parent
        if child.level > self.max_depth:
            self.current.del_child(child)


# use a dummy_trace whenever no tracing is required
dummy_tracer = Tracer(None)


class Node(object):
    """
    A simple tree implementation with labels and associated data.
    """

    def __init__(self, label, parent=None, data=None):
        """Create a new node.

        :param label: some label such as a string or a tuple
        :param parent: nodes know their parents
        :param data: nodes can have associated data which is a key-value store where typically the
            values are statistics
        """

        self.label = label
        if data is None:
            data = OrderedDict()
        self.data = OrderedDict(data)
        self.parent = parent
        self.children = []

    def add_child(self, child):
        """Add a child.

        :param child: a node
        :returns: the child

        """
        child.parent = self
        self.children.append(child)
        return child

    def del_child(self, child):
        """

        >>> root = Node("root")
        >>> c1 = root.child("child1")
        >>> c2 = root.child("child2")
        >>> root.children
        [{"child1": {}}, {"child2": {}}]
        >>> root.del_child(c1)
        >>> root.children
        [{"child2": {}}]

        """
        self.children.remove(child)

    def child(self, label):
        """
        If node has a child labelled ``label`` return it, otherwise add a new child.

        :param label: a label
        :returns: a node

        >>> root = Node("root")
        >>> c1 = root.child("child"); c1
        {"child": {}}
        >>> c2 = root.child("child"); c2
        {"child": {}}
        >>> c1 is c2
        True

        """
        for child in self.children:
            if child.label == label:
                return child

        return self.add_child(Node(label))

    def __str__(self):
        """
        >>> from collections import OrderedDict
        >>> str(Node("root", data=OrderedDict([('a',1), ('b', 2)])))
        '{"root": {"a":        1,  "b":        2}}'
        """
        return u'{"%s": %s}' % (self.label, pretty_dict(self.data))

    __repr__ = __str__

    def report(self, indentation=0, depth=None):
        """
        Return detailed string representation of this tree.

        :param indentation: add spaces to the left of the string representation
        :param depth: stop at this depth

        >>> root = Node("root")
        >>> c1 = root.child(("child",1))
        >>> c2 = root.child(("child",2))
        >>> c3 = c1.child(("child", 3))
        >>> c1.data["a"] = 100.0
        >>> c3.data["a"] = 4097

        >>> print(root.report())
        {"root": {}}
          {"('child', 1)": {"a":  100.000}}
            {"('child', 3)": {"a":     4097}}
          {"('child', 2)": {}}

        >>> print(root.report(indentation=2, depth=1))
          {"root": {}}
            {"('child', 1)": {"a":  100.000}}
            {"('child', 2)": {}}
        """
        s = [" " * indentation + str(self)]
        if depth is None or depth > 0:
            for child in self.children:
                depth = None if depth is None else depth - 1
                s.append(child.report(indentation + 2, depth=depth))
        return "\n".join(s)

    def sum(self, tag, include_self=True, raise_keyerror=False, label=None):
        """
        Return sum over all items tagged ``tag`` in associated data within this tree.

        :param tag: a string
        :param include_self: include data in this node
        :param raise_keyerror: if a node does not have an item tagged with ``tag`` raise a
            ``KeyError``
        :param label: filter by ``label``


        >>> root = Node("root")
        >>> c1 = root.child(("child",1))
        >>> c2 = root.child(("child",2))
        >>> c3 = c1.child(("child", 3))
        >>> c1.data["a"] = 100.0
        >>> c3.data["a"] = 4097

        >>> root.sum("a")
        4197.0

        >>> root.sum("a", label=("child",3))
        4097

        >>> root.sum("a", label=("child",2))
        0

        >>> root.sum("a", label=("child",2), raise_keyerror=True)
        Traceback (most recent call last):
        ...
        KeyError: 'a'

        """
        if include_self and (label is None or self.label == label):
            if raise_keyerror:
                r = self.data[tag]
            else:
                r = self.data.get(tag, 0)
        else:
            r = 0
        for child in self.children:
            r = r + child.sum(tag, include_self=True, raise_keyerror=raise_keyerror, label=label)
        return r

    def find(self, label, raise_keyerror=False):
        """
        Find the first child node matching label in a breadth-first search.

        :param label: a label
        :param raise_keyerror: raise a ``KeyError`` if ``label`` was not found.
        """
        for child in self.children:
            if child.label == label:
                return child
        for child in self.children:
            try:
                return child.find(label, raise_keyerror)
            except KeyError:
                pass

        if raise_keyerror:
            raise KeyError("Label '%s' not present in '%s" % (label, self))
        else:
            return None

    def find_all(self, label):
        """
        Find all nodes labelled ``label``

        :param label: a label

        """
        r = []
        if self.label == label:
            r.append(self)
        if isinstance(self.label, tuple) and self.label[0] == label:
            r.append(self)
        for child in self.children:
            r.extend(child.find_all(label))
        return tuple(r)

    def __iter__(self):
        """
        Depth-first iterate over all subnodes (including self)

        ::

            >>> root = Node("root")
            >>> c1 = root.child(("child",1))
            >>> c2 = root.child(("child",2))
            >>> c3 = c1.child(("child", 3))
            >>> c1.data["a"] = 100.0
            >>> c3.data["a"] = 4097
            >>> list(root)
            [{"root": {}}, {"('child', 1)": {"a":  100.000}}, {"('child', 3)": {"a":     4097}}, {"('child', 2)": {}}]

        """

        yield self
        for child in self.children:
            for c in iter(child):
                yield c

    def merge(self, node):
        """
        Merge tree ``node`` into self.

        .. note :: The label of ``node`` is ignored.
        """

        for k, v in node.data.iteritems():
            if k in self.data:
                self.data[k] += v
            else:
                self.data[k] = v

        for child in node.children:
            self.child(child.label).merge(child)

    def get(self, label):
        """Return first child node with label ``label``

        :param label: label

        >>> root = Node("root")
        >>> _ = root.child("bar")
        >>> c1 = root.child(("foo",0))
        >>> c2 = root.child(("foo",3))
        >>> c3 = c1.child(("foo", 3))
        >>> c1.data["a"] = 100.0
        >>> c3.data["a"] = 4097
        >>> root.get("bar")
        {"bar": {}}

        >>> root.get("foo")
        ({"('foo', 0)": {"a":  100.000}}, {"('foo', 3)": {}})

        >>> root.get("foo")[0]
        {"('foo', 0)": {"a":  100.000}}
        >>> root.get("foo")[1]
        {"('foo', 3)": {}}

        """
        r = []
        for child in self.children:
            if child.label == label:
                return child
            if isinstance(child.label, tuple) and child.label[0] == label:
                r.append(child)
        if r:
            return tuple(r)
        else:
            raise AttributeError("'Node' object has no attribute '%s'" % (label))

    def __getitem__(self, tag):
        """Return associated data tagged ``tag```

        :param tag: Some tag

        >>> root = Node("root", data={"foo": 1})
        >>> c1 = root.child("foo")
        >>> root["foo"]
        1

        """
        return self.data[tag]

    @property
    def level(self):
        """
        Return level of this node, i.e. how many steps it takes to reach a node with parent
        ``None``.

        >>> root = Node("root")
        >>> _ = root.child("bar")
        >>> c1 = root.child(("foo",0))
        >>> c2 = root.child(("foo",3))
        >>> c3 = c1.child(("foo", 3))
        >>> root.level
        0
        >>> c1.level
        1
        >>> c3.level
        2

        """
        node, level = self, 0
        while node.parent is not None:
            level += 1
            node = node.parent
        return level

    def __sub__(self, rhs):
        """
        Return tree that contains the difference of this node and the other.

        The semantics are as follows:

        - For all data in this node the matching data item in ``rhs`` is subtracted.
        - If the data is missing in ``rhs`` it is assumed to be zero.
        - For all children of this node this function is called recursively.
        - If ``rhs`` does not have an immediate child node with a matching label, those children are skipped.
        """

        if not isinstance(rhs, Node):
            raise ValueError("Expected node but got '%s'" % type(rhs))
        diff = Node(self.label)
        for k in self.data:
            diff.data[k] = self.data[k] - rhs.data.get(k, 0)

        for lchild in self.children:
            for rchild in rhs.children:
                if lchild.label == rchild.label:
                    diff.children.append(lchild - rchild)
                    break
            else:
                print("Skipping missing node '%s'" % lchild.label)
        return diff

    def copy(self, deepcopy=True):
        """
        Return a (deep)copy of this node.

        :param deepcopy: If ``False`` child nodes and data dictionaries are not copied, this is
            usually not what the user wants.
        """
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def accumulate(self, key, filter=lambda node: True, repr="avg"):
        """
        Return accumulator value for all occurrences of ``key``::

            >>> root = Node("root")
            >>> c1 = root.child(("child",1))
            >>> c2 = root.child(("child",2))
            >>> c3 = c1.child(("child", 3))
            >>> c1.data["a"] = 100.0
            >>> c3.data["a"] = 4097
            >>> root.accumulate("a").sum
            4197.0
            >>> root.accumulate("a", filter=lambda node: node.label == ("child", 3)).sum
            4097


        :param key: Dictionary key to some ``data`` attribute
        :param filter: Callable that should return ``True`` for nodes that ought to be considered.
        :param repr: Representation of accumulator

        """
        acc = Accumulator(0, repr=repr, count=False)
        for node in iter(self):
            if filter(node) and key in node.data:
                acc += node.data[key]
        return acc


class TimeTreeTracer(Tracer):
    """
    Collect CPU and wall time for every context visited, creating a tree structure along the way.
    """

    def __init__(
        self,
        instance,
        verbosity=False,
        root_label="root",
        start_clocks=False,
        max_depth=1024,
    ):
        """
        Create a new tracer instance.

        :param instance: BKZ-like object instance
        :param verbosity: print information, integers ≥ 0 are also accepted
        :param root_label: label to give to root node
        :param start_clocks: start tracking time for the root node immediately
        :param max_depth: record up to this depth.

        """

        Tracer.__init__(self, instance, verbosity, max_depth)
        self.trace = Node(root_label)
        self.current = self.trace
        if start_clocks:
            self.reenter()

    def enter(self, label, **kwds):
        """Enter new context with label

        :param label: label

        """
        self.current = self.current.child(label)
        self.reenter()

    def reenter(self, **kwds):
        """Reenter current context, i.e. restart clocks"""

        if self.current is None:
            # we exited the root node
            self.current = self.trace
        node = self.current
        node.data["cputime"] = node.data.get("cputime", 0) + Accumulator(
            -process_time(), repr="sum", count=False
        )
        node.data["walltime"] = node.data.get("walltime", 0) + Accumulator(
            -time.time(), repr="sum", count=False
        )

    def exit(self, **kwds):
        """
        Leave context, record time spent.

        :param label: ignored

        .. note :: If verbosity ≥ to the current level, also print the current node.

        """
        node = self.current

        node.data["cputime"] += process_time()
        node.data["walltime"] += time.time()

        if self.verbosity and self.verbosity >= self.current.level:
            print(self.current)

        self._pop()


class BKZTreeTracer(Tracer):
    """
    Default tracer for BKZ-like algorithms.
    """

    def __init__(
        self, instance, verbosity=False, root_label="bkz", start_clocks=False, max_depth=16
    ):
        """
        Create a new tracer instance.

        :param instance: BKZ-like object instance
        :param verbosity: print information, integers ≥ 0 are also accepted
        :param root_label: label to give to root node
        :param start_clocks: start tracking time for the root node immediately
        :param max_depth: record up to this depth.

        TESTS::

            >>> from fpylll.tools.bkz_stats import BKZTreeTracer
            >>> tracer = BKZTreeTracer(None)
            >>> for i in range(3): tracer.enter("level-%d"%i)
            >>> for i in range(3): tracer.exit()
            >>> "level-2" in tracer.trace.report()
            True
            >>> tracer = BKZTreeTracer(None, max_depth=2)
            >>> for i in range(3): tracer.enter("level-%d"%i)
            >>> for i in range(3): tracer.exit()
            >>> "level-2" in tracer.trace.report()
            False

        """

        Tracer.__init__(self, instance, verbosity, max_depth)
        self.trace = Node(root_label)
        self.current = self.trace
        if start_clocks:
            self.reenter()

    def enter(self, label, **kwds):
        """Enter new context with label

        :param label: label

        """
        self.current = self.current.child(label)
        self.reenter()

    def reenter(self, **kwds):
        """Reenter current context, i.e. restart clocks"""

        node = self.current
        node.data["cputime"] = node.data.get("cputime", 0) + Accumulator(
            -process_time(), repr="sum", count=False
        )
        node.data["walltime"] = node.data.get("walltime", 0) + Accumulator(
            -time.time(), repr="sum", count=False
        )

    def exit(self, **kwds):  # noqa, shut up linter about this function being too complex
        """
        By default CPU and wall time are recorded.  More information is recorded for "enumeration"
        and "tour" labels.  When the label is a tour then the status is printed if verbosity > 0.
        """
        node = self.current
        label = node.label

        node.data["cputime"] += process_time()
        node.data["walltime"] += time.time()

        if kwds.get("dump_gso", False):
            node.data["r"] = node.data.get("r", []) + [self.instance.M.r()]

        if label == "enumeration":
            full = kwds.get("full", True)
            if full:
                try:
                    node.data["#enum"] = Accumulator(
                        kwds["enum_obj"].get_nodes(), repr="sum"
                    ) + node.data.get(
                        "#enum", None
                    )  # noqa
                except KeyError:
                    pass
                try:
                    node.data["%"] = Accumulator(kwds["probability"], repr="avg") + node.data.get(
                        "%", None
                    )
                except KeyError:
                    pass

        if label[0] == "tour":
            data = basis_quality(self.instance.M)
            for k, v in data.items():
                if k == "/":
                    node.data[k] = Accumulator(v, repr="max")
                else:
                    node.data[k] = Accumulator(v, repr="min")

        if self.verbosity and label[0] == "tour":
            report = OrderedDict()
            report["i"] = label[1]
            report["cputime"] = node["cputime"]
            report["walltime"] = node["walltime"]
            try:
                report["preproc"] = node.find("preprocessing", True)["cputime"]
            except KeyError:
                pass
            try:
                report["svp"] = node.find("enumeration", True)["cputime"]
            except KeyError:
                pass
            report["#enum"] = node.sum("#enum")
            report["lll"] = node.sum("cputime", label="lll")
            try:
                report["pruner"] = node.find("pruner", True)["cputime"]
            except KeyError:
                pass
            report["r_0"] = node["r_0"]
            report["/"] = node["/"]

            print(pretty_dict(report))

        self._pop()


def normalize_tracer(tracer):
    """
    Normalize tracer inputs for convenience.

    :param tracer:  ``True`` for ``BKZTreeTracer``, ``False`` for ``dummy_tracer``
                     or any other value for custom tracer.

    EXAMPLE::

        >>> from fpylll.tools.bkz_stats import normalize_tracer, BKZTreeTracer, dummy_tracer
        >>> normalize_tracer(True) == BKZTreeTracer
        True
        >>> normalize_tracer(False) == dummy_tracer
        True
        >>> normalize_tracer(BKZTreeTracer) == BKZTreeTracer
        True

    """
    if tracer is True:
        return BKZTreeTracer
    elif tracer is False:
        return dummy_tracer
    else:
        return tracer
