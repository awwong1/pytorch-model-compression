import matplotlib.pyplot as plt
import numpy as np


class Scribe(object):
    """Save model training process to a log file with a simple plot function.
    Based off of the Logger class from
    https://github.com/bearpaw/pytorch-classification/blob/24f1c456f48c78133088c4eefd182ca9e6199b03/utils/logger.py#L23
    """

    def __init__(self, filepath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if not title else title
        if filepath:
            if resume:
                self.file = open(filepath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(filepath, "a")
            else:
                self.file = open(filepath, "w")

    def set_names(self, names):
        if self.resume:
            return
        self.numbers = {}
        self.names = names
        for name in names:
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Mismatched values length"
        for idx, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[idx]].append(num)
        self.file.write("\n")
        self.file.flush()

    def plot(self, plot_title=None, names=None, xlabel=None, ylabel=None):
        names = self.names if names is None else names
        numbers = self.numbers
        for name in names:
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([f"{self.title}({name})" for name in names])
        plt.grid(True)
        if plot_title is not None:
            plt.title(plot_title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

    def close(self):
        if self.file:
            self.file.close()

    @staticmethod
    def savefig(fname, dpi=None):
        if dpi is None:
            dpi = 150
        plt.savefig(fname, dpi=dpi)
        plt.clf()
