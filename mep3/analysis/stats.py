import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Stats():
    def __init__(self):
        self.to_save = True

    @staticmethod
    def 

class PlotStats():
    def __init__(self):
        self.to_save = True

    @staticmethod
    def plot_histogram(data, bins=10, title=None, xlabel=None, ylabel=None):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.grid(axis='y', alpha=0.75)
        plt.show()