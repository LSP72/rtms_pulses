import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ProcessingPerPP():
    def __init__(self):
        self.to_save = True

    @staticmethod
    def average(data):
        """
        Computes the average of a given data array.

        Parameters
        ----------
        data : array-like
            The input data array.
        
        Returns
        -------
        float
            The average of the input data.
        """
        if type(data) is pd.DataFrame or type(data) is pd.Series:
            data = data.values.flatten()
        return np.mean(data)

    @staticmethod
    def std(data):
        """
        Computes the standard deviation of a given data array.

        Parameters
        ----------
        data : array-like
            The input data array.
        
        Returns
        -------
        float
            The standard deviation of the input data.
        """
        if type(data) is pd.DataFrame or type(data) is pd.Series:
            data = data.values.flatten()
        return np.std(data)

    @staticmethod
    def find_mwave_max(mwaves: pd.Series) -> tuple:
        """
        Finds the index of the maximum value in the MWAVES data.

        Parameters
        ----------
        mwaves : pd.Series
            The MWAVES data.
        
        Returns
        -------
        tuple
            A tuple containing the maximum value and its index in the MWAVES data.
        """
        return [mwaves.max(), mwaves.idxmax()]


class Processing():
    @staticmethod
    def mwave_normalisation(meps: pd.Series, mwaves: pd.Series, percentage: bool = False) -> pd.Series:
        """
        Normalizes the MEPS data based on the MWAVES data.

        Parameters
        ----------
        meps : pd.Series
            The MEPS data to be normalized.
        mwaves : pd.Series
            The MWAVES data used for normalization.
        
        Returns
        -------
        pd.Series
            The normalized MEPS data.
        """
        if percentage:
            return (meps / mwaves) * 100
        else:
            return meps / mwaves

    @staticmethod
    def pre_normalisation(meps: pd.Series, pre: pd.Series, percentage: bool = False) -> pd.Series:
        """
        Normalizes the MEPS data based on the PRE data.

        Parameters
        ----------
        meps : pd.Series
            The MEPS data to be normalized.
        pre : pd.Series
            The PRE data used for normalization.
        
        Returns
        -------
        pd.Series
            The normalized MEPS data.
        """
        if percentage:
            return (meps / pre) * 100
        else:
            return meps / pre
    