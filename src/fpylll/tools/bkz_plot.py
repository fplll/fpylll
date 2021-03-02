# -*- coding: utf-8 -*-
"""
Plot `\\log_2` of square Gram-Schmidt norms during a BKZ run.

EXAMPLE::

    >>> from fpylll import IntegerMatrix, BKZ, FPLLL
    >>> from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
    >>> from fpylll.tools.bkz_plot import KeepGSOBKZFactory
    >>> FPLLL.set_random_seed(1337)
    >>> _ = FPLLL.set_threads(1)  # to make it deterministic
    >>> A = IntegerMatrix.random(80, "qary", k=40, bits=20)
    >>> bkz = KeepGSOBKZFactory(BKZ2)(A)
    >>> bkz(BKZ.EasyParam(20))
    >>> bkz._KeepGSOBKZ__gso_norms[0][0]
    23104295.0
    >>> bkz._KeepGSOBKZ__gso_norms[-1][0]
    6591824.0

.. modulauthor: Martin Albrecht <martin.albrecht@royalholloway.ac.uk>

"""


def KeepGSOBKZFactory(cls):
    """
    Return a wrapper class around ``cls`` which collects Gram-Schmidt norms in the attribute
    ``__KeepGSOBKZ_gso_norms``.

    In particular, the list will be constructed as follows:

    - index 0: input GSO norms
    - index i,j: kappa and GSO norms in tour i-1 for after j-th SVP call
    - index -1: output GSO norms

    :param cls: A BKZ-like algorithm with methods ``__call__``, ``svp_reduction`` and ``tour``.

    .. warning:: This will slow down the algorithm especially for small block sizes.

    """
    class KeepGSOBKZ(cls):
        def __call__(self, *args, **kwds):
            self.M.update_gso()
            self.__gso_norms = [self.M.r()]
            self.__at_toplevel = True
            cls.__call__(self, *args, **kwds)
            self.M.update_gso()
            self.__gso_norms.append(self.M.r())

        def svp_reduction(self, kappa, *args, **kwds):
            at_toplevel = self.__at_toplevel
            self.__at_toplevel = False
            r = cls.svp_reduction(self, kappa, *args, **kwds)
            self.__at_toplevel = at_toplevel
            if at_toplevel:
                self.M.update_gso()
                self.__gso_norms[-1].append((kappa, self.M.r()))
            return r

        def tour(self, *args, **kwds):
            if self.__at_toplevel:
                self.__gso_norms.append([])
            return cls.tour(self, *args, **kwds)

    return KeepGSOBKZ


def plot_gso_norms(gso_norms, block_size, basename="bkz-gso-norms",
                   extension="png", dpi=300):
    """Plot ``gso_norms``.

    :param gso_norms: list of GSO norms. It is assumed these follow the form output by ``KeepGSOBKZ``.
    :param block_size: BKZ block size
    :param basename: graphics filename basenname (may contain full path)
    :param extension: graphics filename extension/type
    :param dpi: resolution

    :returns: Tuple of filenames written.

    .. note:: To convert to movie, call e.g. ``ffmpeg -framerate 8 -pattern_type glob -i "*.png" bkz.mkv``

    .. warning:: This function is quite slow.
    """
    from math import log, pi, e
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    filenames = []

    def maplog2(l):
        return [log(l[i], 2) for i in range(len(l))]

    def plot_finalize(ax, name):
        ax.set_ylabel("$2\\,\\log_2(\\cdot)$")
        ax.set_xlabel("$i$")
        ax.legend(loc="upper right")
        ax.set_ylim(*ylim)

        fullname = "%s.%s"%(name, extension)
        fig.savefig(fullname, dpi=dpi)
        filenames.append(fullname)
        plt.close()

    d = len(gso_norms[0])
    x = range(d)

    beta = float(block_size)
    delta_0 = (beta/(2.*pi*e) * (pi*beta)**(1./beta))**(1./(2.*(beta-1)))
    alpha = delta_0**(-2.*d/(d-1.))
    logvol = sum(maplog2(gso_norms[0]))  # already squared
    gsa = [log(alpha, 2)*(2*i) + log(delta_0, 2)*(2*d) + logvol*(1./d) for i in range(d)]

    fig, ax = plt.subplots()
    ax.plot(x, maplog2(gso_norms[0]), label="$\\|\\mathbf{b}_i^*\\|$")
    ylim = ax.get_ylim()
    ax.set_title("Input")
    plot_finalize(ax, "%s-aaaa-input"%basename)

    for i, tour in enumerate(gso_norms[1:-1]):
        for j, (kappa, norms) in enumerate(tour):
            fig, ax = plt.subplots()

            rect = patches.Rectangle((kappa, ylim[0]), min(block_size, d-kappa-1), ylim[1]-ylim[0],
                                     fill=True, color="lightgray")
            ax.add_patch(rect)
            ax.plot(x, maplog2(norms), label="$\\|\\mathbf{b}_i^*\\|$")
            ax.plot(x, gsa, color="black", label="GSA")
            ax.set_title("BKZ-%d tour: %2d, $\\kappa$: %3d"%(block_size, i, kappa))
            plot_finalize(ax, "%s-t%03d-%04d"%(basename, i, j))

    fig, ax = plt.subplots()
    ax.plot(x, maplog2(gso_norms[-1]))
    ax.set_title("Output")
    plot_finalize(ax, "%s-zzzz-output"%basename)

    return tuple(filenames)
