"""
nilin
"""


import numpy as np
import math
import sys
import os
import argparse


def ndigits(x):
    return math.floor(1 + math.log(x) / math.log(10))


class Figure:
    plots = []
    labels = []
    xscale: str = "linear"
    yscale: str = "linear"
    xlim = (None, None)
    ylim = (None, None)
    hori_pad: int = 20
    vert_pad: int = 10

    def __init__(self, h=None, w=None):
        if h is None:
            h = os.get_terminal_size().lines - 2 * self.vert_pad
        if w is None:
            w = os.get_terminal_size().columns - 2 * self.hori_pad
        self.h = h
        self.w = w
        self.grid = [[" " for j in range(w)] for i in range(h)]
        self.lbar = ["" for _ in range(h)]

    def getpos(self, x, y, xlim, ylim):
        a, b = xlim
        c, d = ylim
        j = self.w * (x - a) / (b - a)
        i = self.h - self.h * (y - c) / (d - c)
        return round(i), round(j), i - round(i), j - round(j)

    def plot(self, X, Y=None, style="quarter-block", label=""):
        if Y is None:
            X, Y = np.arange(len(X)), X

        X = self.downsample(X)
        Y = self.downsample(Y)

        self.plots.append((X, Y, style))
        self.labels.append(label)
        self.update_lims(X, Y)

    def renderplot(self, X, Y, style):
        if style == "quarter-block":
            styler = QuarterBlockStyler()
        else:
            styler = Styler(style)

        sparse_plot = self.get_sparse_plot(X, Y, styler)
        for (i, j), pixel in sparse_plot:
            if i >= 0 and i < self.h and j >= 0 and j < self.w:
                self.grid[i][j] = styler.update_pixel(pixel, self.grid[i][j])

    def get_sparse_plot(self, X, Y, styler):
        out = []
        for x, y in zip(X, Y):
            i, j, di, dj = self.getpos(*self.transform(x, y, self.xlim, self.ylim))
            out.append(((i, j), styler.get_initial_pixel(di, dj)))
        return out

    def render_y_ticks(self):
        yticks = self.get_yticks()
        labels = [f"{y:.3f} -- " for y in yticks]
        barwidth = self.hori_pad
        self.lbar = [" " * barwidth for _ in range(self.h)]

        for y, label in zip(yticks, labels):
            i, _, t, _ = self.getpos(*self.transform(0, y, (0, 1), self.ylim))
            if i >= 0 and i < self.h:
                self.lbar[i] = " " * (barwidth - len(label)) + label[-barwidth:]

    def show(self):
        for X, Y, style in self.plots:
            self.renderplot(X, Y, style)

        self.render_y_ticks()

        rows = [label + "".join(row) for label, row in zip(self.lbar, self.grid)]
        img = ("\n").join(rows)

        print(self.legend())
        print(img)
        print("\n" * self.vert_pad, end="")

    def get_yticks(self):
        if self.yscale == "log":
            ticks = self.get_log_ticks(*self.ylim)
        else:
            ticks = self.get_linear_ticks(*self.ylim)
        return ticks

    def update_lims(self, X, Y):
        a, b = self.xlim
        c, d = self.ylim

        self.xlim = (self.min(a, min(X)), self.max(b, max(X)))
        self.ylim = (
            self.min(c, min(Y) * 1.2 - max(Y) * 0.2),
            self.max(d, max(Y) * 1.2 - min(Y) * 0.2),
        )

    def transform(self, x, y, xlim, ylim):
        if self.xscale == "log":
            x = np.log(x)
            xlim = (np.log(xlim[0]), np.log(xlim[1]))
        if self.yscale == "log":
            y = np.log(y)
            ylim = (np.log(ylim[0]), np.log(ylim[1]))
        return x, y, xlim, ylim

    def legend(self):
        rows = []
        for (X, Y, style), label in zip(self.plots, self.labels):
            if style == "quarter-block":
                style = QuarterBlockStyler.diag
            rows.append(self.hori_pad * " " + f"{style} {label}")
        rows.append("")
        if len(rows) < self.vert_pad:
            rows = [""] * (self.vert_pad - len(rows)) + rows
        return "\n".join(rows)

    @staticmethod
    def downsample(x):
        resolution = 1000
        if len(x) < 5 * resolution:
            return x

        blocksize = len(x) // resolution
        cutoff = resolution * blocksize
        x = np.reshape(x[:cutoff], (resolution, blocksize))
        x = np.mean(x, axis=1)
        return x

    @staticmethod
    def max(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    @staticmethod
    def min(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    @staticmethod
    def get_linear_ticks(a, b, orderdifference=2):
        step = 10 ** (ndigits(b - a) - orderdifference)
        a = math.floor(a / step) * step
        b = math.ceil(b / step) * step
        return np.arange(a, b, step)

    @staticmethod
    def get_log_ticks(a, b):
        p1 = ndigits(a)
        p2 = ndigits(b)
        return [10**p for p in range(p1, p2 + 1)]


class Styler:
    def __init__(self, style=None):
        self.style = style

    def get_initial_pixel(self, di, dj):
        return self.style

    def update_pixel(self, new, prev):
        return new


class QuarterBlockStyler(Styler):
    lowerhalfblock = "\u2584"
    upperhalfblock = "\u2580"
    lefthalfblock = "\u258C"
    righthalfblock = "\u2590"
    block = "\u2588"
    nw = "\u2598"
    ne = "\u259D"
    sw = "\u2596"
    se = "\u2597"
    diag = "\u259A"
    offdiag = "\u259E"

    quarterblocktable = {
        (0, 0, 0, 0): " ",
        (0, 0, 0, 1): se,
        (0, 0, 1, 0): sw,
        (0, 0, 1, 1): lowerhalfblock,
        (0, 1, 0, 0): ne,
        (0, 1, 0, 1): righthalfblock,
        (0, 1, 1, 0): offdiag,
        (0, 1, 1, 1): "\u259F",
        (1, 0, 0, 0): nw,
        (1, 0, 0, 1): diag,
        (1, 0, 1, 0): lefthalfblock,
        (1, 0, 1, 1): "\u2599",
        (1, 1, 0, 0): upperhalfblock,
        (1, 1, 0, 1): "\u259C",
        (1, 1, 1, 0): "\u259B",
        (1, 1, 1, 1): block,
    }

    def __init__(self):
        self.quarterblock_keys = {c: k for k, c in self.quarterblocktable.items()}

    def get_initial_pixel(self, di, dj):
        south = di > 0
        east = dj > 0
        return [[self.nw, self.ne], [self.sw, self.se]][south][east]

    def update_pixel(self, new, prev):
        if prev not in self.quarterblock_keys:
            return new

        def _max(a, b):
            return tuple([max(x, y) for x, y in zip(a, b)])

        keys = self.quarterblock_keys

        return self.quarterblocktable[_max(keys[new], keys[prev])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--files", nargs="*", action="store")
    args = parser.parse_args()

    def test():
        x = np.arange(-1, 1, 0.001)
        y = x
        z = 16 * x**5 - 20 * x**3 + 5 * x

        fig = Figure()
        fig.plot(y, style="*", label="z")
        fig.plot(z, label="y", style="quarter-block")

        if args.logy:
            fig.yscale = "log"

        if args.ylim is not None:
            fig.ylim = args.ylim

        fig.show()
        fig.legend()

    if args.files is None:
        test()
    else:
        fig = Figure()
        paths = [path for path in args.files if os.path.isfile(path)]
        styles = ["quarter-block", "*", "#", "o"]
        for path, style in zip(paths, styles):
            data = np.loadtxt(path)

            if len(data.shape) == 1:
                fig.plot(data, style=style, label=path)
            elif len(data.shape) == 2:
                fig.plot(data[:, 0], data[:, 1], style=style, label=path)
            else:
                raise ValueError("data must have 1 or 2 columns")

        if args.logy:
            fig.yscale = "log"

        if args.ylim is not None:
            fig.ylim = args.ylim

        fig.show()
        fig.legend()
