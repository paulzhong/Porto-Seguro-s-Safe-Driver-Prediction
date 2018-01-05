# https://github.com/ResidentMario/missingno
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
# import seaborn as sns


class MissingValue(object):
    """
        handeling missing value

    """

#     def __init__(self, dataframe):
#         self.dataframe = dataframe

    def DesMissingValue(self, dataframe, missing=np.NaN):
        vars_with_missing = []

        for f in dataframe.columns:
            missings = dataframe[dataframe[f] == missing][f].count()
            if missings > 0:
                vars_with_missing.append(f)
                missings_perc = missings/dataframe.shape[0]

                print('Variable {} has {} records ({:.2%}) with missing values'.format(f,
                                                                                       missings,
                                                                                       missings_perc))

        print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))

    def Replace(self, dataframe, old, new):
        dataframe = dataframe.replace(old, new)
        return dataframe

    def PlotMissingMatrix(self, df, start, end):
        """
        input:
            df: dataframe
            start: the column of dataframe we want to start
            end: the column of dataframe we want to end

        """
        plotMM = msno.matrix(df=df.iloc[:, start:end],
                   figsize=(20, 14), color=(0.7, 0.01, 0.01))
        plt.show(plotMM)

    def PlotMissingBar(self, df, start, end):
        """
        input:
            df: dataframe
            start: the column of dataframe we want to start
            end: the column of dataframe we want to end

        """
        plotMB = msno.bar(df.iloc[:, start:end], figsize=(20, 14))
        plt.show(plotMB)

    def PlotMissingHeatMap(self, df, start, end):
        """
            The missingno correlation heatmap measures nullity correlation:
            how strongly the presence or absence of one variable affects
            the presence of another.

        """
        plotHM = msno.heatmap(df.iloc[:, start:end], figsize=(20, 14))
        plt.show(plotHM)
