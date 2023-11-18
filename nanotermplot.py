import numpy as np
import math
import os
import argparse


def ndigits(x):
    return math.floor(1 + math.log(x) / math.log(10))


class Figure:
    plots = []
    labels = []
    xscale: str = "linear"
    yscale: str = "linear"
    xlim = None
    ylim = None
    hori_pad: int = 30
    vert_pad: int = 10
    lim_mode: str = "quantile"

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

        self.plots.append((X, Y, style))
        self.labels.append(label)

    def renderplot(self, X, Y, style):
        if style == "quarter-block":
            styler = QuarterBlockStyler()
        else:
            styler = Styler(style)

        X, Y = self.downsample(X, Y)
        sparse_plot = self.get_sparse_plot(X, Y, styler)
        for (i, j), pixel in sparse_plot:
            if i >= 0 and i < self.h and j >= 0 and j < self.w:
                self.grid[i][j] = styler.update_pixel(pixel, self.grid[i][j])

    def thinning_and_connect(self, X, Y, xlim, ylim, eps=0.002):
        XY = np.stack([X, Y], axis=1)
        p0 = XY[0]
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        metric = np.array([dx**2, dy**2])
        XY_out = [p0[None, :]]
        for p in XY[1:]:
            reldist = np.sum((p - p0) ** 2 / metric)
            if reldist > eps**2:
                t_ = np.arange(0, 1, max(0.001, eps**2 / reldist))
                XY_out.append(p0[None, :] + t_[:, None] * (p - p0)[None, :])
                XY_out.append(p[None, :])
                p0 = p
        XY_out = np.concatenate(XY_out, axis=0)
        return XY_out[:, 0], XY_out[:, 1]

    def get_sparse_plot(self, X, Y, styler):
        out = []
        xlim = self.transform_x(self.xlim)
        ylim = self.transform_y(self.ylim)
        X = self.transform_x(X)
        Y = self.transform_y(Y)
        indices = np.where(np.isfinite(X) * np.isfinite(Y))
        X = X[indices]
        Y = Y[indices]

        X, Y = self.thinning_and_connect(X, Y, xlim, ylim)

        for x, y in zip(X, Y):
            i, j, di, dj = self.getpos(x, y, xlim, ylim)
            out.append(((i, j), styler.get_initial_pixel(di, dj)))
        return out

    def transform_x(self, X):
        X = np.array(X)
        if self.xscale == "log":
            return np.log(X)
        else:
            return X

    def transform_y(self, Y):
        Y = np.array(Y)
        if self.yscale == "log":
            return np.log(Y)
        else:
            return Y

    def render_y_ticks(self):
        yticks = self.get_yticks()
        labels = [f"{y}" for y in yticks]
        barwidth = self.hori_pad
        self.lbar = [" " * barwidth for _ in range(self.h)]

        ylim = self.transform_y(self.ylim)
        yticks = self.transform_y(yticks)

        for y, label in zip(yticks, labels):
            i, _, t, _ = self.getpos(0, y, (0, 1), ylim)

            if i >= 0 and i < self.h:
                label += " " + HorizontalStyler.get_hline(t) * 3 + " "
                self.lbar[i] = " " * (barwidth - len(label)) + label[-barwidth:]

    def show(self):
        self.set_lims()
        for X, Y, style in self.plots:
            self.renderplot(X, Y, style)

        self.render_y_ticks()

        rows = [label + "".join(row) for label, row in zip(self.lbar, self.grid)]
        img = ("\n").join(rows)

        framedimg = "\n".join([self.legend(), img, "\n" * self.vert_pad])
        print(framedimg)
        return framedimg

    def get_ticks(self, lim, scale):
        if scale == "log":
            ticks = self.get_log_ticks(*lim)
        else:
            ticks = self.get_linear_ticks(*lim)
        return ticks

    def get_yticks(self):
        return self.get_ticks(self.ylim, self.yscale)

    def get_xticks(self):
        return self.get_ticks(self.xlim, self.xscale)

    def get_lim(self, Xs, margin=0.2, lim_mode="quantile", scale="linear", lim=None):
        if lim is not None:
            return lim

        if lim_mode == "max":
            x_max = max([max(X) for X in Xs])
            x_min = min([min(X) for X in Xs])
        if lim_mode == "quantile":
            x_max = max([np.quantile(X, q=0.995) for X in Xs])
            x_min = min([np.quantile(X, q=0.005) for X in Xs])

        xlim = np.array([x_min, x_max])
        xlim = (1 + margin) * xlim - margin * np.mean(xlim)

        if scale == "log":
            xlim = np.maximum(xlim, 1e-5)

        return xlim

    def set_lims(self):
        """If one limit is set, we restrict the data accordingly
        to calcule the other limit."""

        if self.xlim is not None:
            self.restrict_x()
        if self.ylim is not None:
            self.restrict_y()

        self.xlim = self.get_lim(
            [plot[0] for plot in self.plots], lim=self.xlim, scale=self.xscale
        )
        self.ylim = self.get_lim(
            [plot[1] for plot in self.plots], lim=self.ylim, scale=self.yscale
        )
        self.restrict_x()
        self.restrict_y()

    def restrict(self, *Xs, lim, axis=0):
        X = Xs[axis]
        indices = np.where((X >= lim[0]) * (X <= lim[1]))
        return [X[indices] for X in Xs]

    def restrict_x(self):
        self.plots = [
            (*self.restrict(X, Y, lim=self.xlim, axis=0), style)
            for X, Y, style in self.plots
        ]

    def restrict_y(self):
        self.plots = [
            (*self.restrict(X, Y, lim=self.ylim, axis=1), style)
            for X, Y, style in self.plots
        ]

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
    def downsample(x, y):
        resolution = 1000
        if len(x) < 5 * resolution:
            return x, y

        blocksize = len(x) // resolution
        cutoff = resolution * blocksize
        x = np.reshape(x[:cutoff], (resolution, blocksize))
        x = np.mean(x, axis=1)
        y = np.reshape(y[:cutoff], (resolution, blocksize))
        y = np.mean(y, axis=1)
        return x, y

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
    def get_linear_ticks(a, b):
        step = 10 ** (ndigits(b - a) - 1)

        if b - a < 2 * step:
            step /= 5
        elif b - a < 5 * step:
            step /= 2

        factor = 10 ** (ndigits(step) - 1)
        step = round(step / factor) * factor

        a = math.floor(a / step) * step
        b = math.ceil(b / step) * step
        return np.arange(a, b, step)

    @staticmethod
    def get_log_ticks(a, b):
        a = max(a, 1e-10)
        p1 = ndigits(a)
        p2 = ndigits(b)
        tenticks = [10**p for p in range(p1, p2)]
        return [x for n in tenticks for x in [n, 2 * n, 5 * n]]


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


class HorizontalStyler(Styler):
    def get_initial_pixel(self, di, dj):
        return self.get_hline(di)

    @staticmethod
    def get_hline(di):
        lowerhalfblock = "\u2584"
        upperhalfblock = "\u2580"
        if di > 0:
            return lowerhalfblock
        else:
            return upperhalfblock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--files", nargs="*", action="store")
    args = parser.parse_args()

    def test(fig):
        x = np.arange(-1, 1, 0.001)
        y = x
        z = 16 * x**5 - 20 * x**3 + 5 * x
        fig.plot(x, z, label="y", style="quarter-block")
        fig.plot(x, y, style="#", label="z")

    fig = Figure()

    if args.files is None:
        test(fig)
    else:
        fig = Figure()
        paths = [path for path in args.files if os.path.isfile(path)]
        styles = ["quarter-block", "#", "*", "o"]
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

    if args.logx:
        fig.xscale = "log"

    if args.ylim is not None:
        fig.ylim = args.ylim

    if args.xlim is not None:
        fig.xlim = args.xlim

    fig.show()
    fig.legend()
