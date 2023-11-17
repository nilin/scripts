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
    xscale: str = "linear"
    yscale: str = "linear"
    xlim = (None, None)
    ylim = (None, None)
    leftpad: int = 10
    vertpad: int = 5
    labels = []

    def __init__(self, h=50, w=os.get_terminal_size().columns - 40):
        self.h = h
        self.w = w
        self.grid = [[" " for j in range(w)] for i in range(h)]
        self.lbar = ["" for _ in range(h)]

    def getpos(self, x, y, xlim, ylim):
        a, b = xlim
        c, d = ylim
        j = self.w * (x - a) / (b - a)
        i = self.h - self.h * (y - c) / (d - c)
        return round(i), round(j)

    def plot(self, X, Y=None, style="*", label=""):
        if Y is None:
            X, Y = np.arange(len(X)), X

        X = self.downsample(X)
        Y = self.downsample(Y)

        self.plots.append((X, Y, style))
        self.labels.append(label)
        self.update_lims(X, Y)

    def renderplot(self, X, Y, style):
        for x, y in zip(X, Y):
            i, j = self.getpos(*self.transform(x, y, self.xlim, self.ylim))
            if i >= 0 and i < self.h and j >= 0 and j < self.w:
                self.grid[i][j] = style

    def render_y_ticks(self):
        yticks = self.get_yticks()
        labels = [f"{y:.3f}" for y in yticks]
        barwidth = max([len(label) for label in labels])
        self.lbar = [" " * barwidth for _ in range(self.h)]

        for y, label in zip(yticks, labels):
            i, _ = self.getpos(*self.transform(0, y, (0, 1), self.ylim))
            if i >= 0 and i < self.h:
                self.lbar[i] = " " * (barwidth - len(label)) + label

    def show(self):
        for X, Y, style in self.plots:
            self.renderplot(X, Y, style)

        self.render_y_ticks()

        rows = [
            self.leftpad * " " + label + "".join(row)
            for label, row in zip(self.lbar, self.grid)
        ]
        img = ("\n").join(rows)
        img = "\n" * self.vertpad + img + "\n" * self.vertpad
        print(img)

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
        for (X, Y, style), label in zip(self.plots, self.labels):
            print(self.leftpad * " ", style, label)

    @staticmethod
    def downsample(x):
        resolution = 200
        if len(x) < resolution:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--files", nargs="*", action="store")
    args = parser.parse_args()

    def test():
        x = np.arange(100)
        y = x**2 + 1
        z = 50 * x + 1

        fig = Figure()
        fig.plot(y, label="y")
        fig.plot(z, style="-", label="z")

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
        styles = ["*", "#", "o", "-"]
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
