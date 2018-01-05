import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TrainTestAnalysis(object):
    """
        Input: trian, test

    """

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def TrainTestCompare(self):

        print('We have {} training rows and {} test rows.'.format(self.train.shape[0], self.test.shape[0]))
        print('We have {} training columns and {} test columns.'.format(self.train.shape[1], self.test.shape[1]))

        if len(np.intersect1d(self.train.id.values, self.test.id.values)) == 0:
            print('Train and test sets are distinct.')
        else:
            print('oops! Train and test sets are NOT distinct.')

        if self.train.count().min() == self.train.shape[0] and self.test.count().min() == self.test.shape[0]:
            print('There is no NaN')
        else:
            print('oops! Missing value detected')

    def PlotTrainTestMissing(self, old=0, new=0):

        if (new is np.nan):

            traint = self.train.replace(old, new)
            testt = self.test.replace(old, new)
            train_test_missing = pd.DataFrame()
            train_test_missing['train'] = traint.isnull().sum()
            train_test_missing['test'] = testt.isnull().sum()
            fig, ax = plt.subplots(figsize=(20, 14))
            train_test_missing.plot(kind='bar', ax=ax)
            plt.show(fig)

        else:

            traint = self.train
            testt = self.test
            train_test_missing = pd.DataFrame()
            train_test_missing['train'] = traint.isnull().sum()
            train_test_missing['test'] = testt.isnull().sum()
            fig, ax = plt.subplots(figsize=(20, 14))
            train_test_missing.plot(kind='bar', ax=ax)
            plt.show(fig)
