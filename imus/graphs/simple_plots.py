import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plot():
    def __init__(self):
        self.xlen = None
        self.x = None
        self.fs = None
        self.to_save = True
        

    def same_plot_3_axes(self, sig_x, sig_y, sig_z, lim=None, type=None):
        
        fig = plt.figure(figsize=(20, 8))

        frame_len = len(self.x)//4
        plt.plot(self.x[0:frame_len], sig_x[0:frame_len], label="x", alpha=0.5, color="blue")
        plt.plot(self.x[0:frame_len], sig_y[0:frame_len], label="y", alpha=0.5, color="violet")
        plt.plot(self.x[0:frame_len], sig_z[0:frame_len], label="z", alpha=0.5, color="darkmagenta")
        plt.xlabel("Time (s)")
        if type is not None:

            plt.title("Gyrometer signals")
            plt.ylabel("Angular velocity (°/s)")
        if lim is not None:
            plt.xlim(lim)
        plt.legend()
        plt.grid()
        plt.show()
