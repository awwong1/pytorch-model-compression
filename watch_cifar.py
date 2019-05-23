#!/usr/bin/env python3
"""CIFAR-10 and CIFAR-100 tensorwatch visualizer
"""
import tensorwatch as tw
import time


def main():
    w = tw.Watcher(filename="tw_test.log")
    s = w.create_stream(name="my_metric")
    w.make_notebook()

    for i in range(1000):
        s.write((i, i * i))
        time.sleep(1)


if __name__ == "__main__":
    main()
