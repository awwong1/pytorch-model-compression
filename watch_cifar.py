#!/usr/bin/env python3
"""CIFAR-10 and CIFAR-100 tensorwatch visualizer
"""
import tensorwatch as tw
import time
from cifar import initialize_model

def main():
    # work in progress
    w = tw.Watcher(filename="tw_test.log")
    s = w.create_stream(name="my_metric")
    model = initialize_model("alexnet", num_classes=10)
    w.observe(model)
    w.make_notebook()

    for i in range(1000):
        s.write((i, i * i))
        time.sleep(1)


if __name__ == "__main__":
    main()
