"""
Categorical Variables:
A categorical or discrete variable is one that has two or more categories
(values). There are two types of categorical variable, nominal and ordinal.
A nominal variable has no intrinsic ordering to its categories. For example,
gender is a categorical variable having two categories (male and female) with
 no intrinsic ordering to the categories. An ordinal variable has a clear
 ordering. For example, temperature as a variable with three orderly categories
(low, medium and high). A frequency table is a way of counting how often each
category of the variable in question occurs. It may be enhanced by the
addition of percentages that fall into each category.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import collections

from scipy.stats import kurtosis
from scipy.stats import skew

from scipy import stats

py.init_notebook_mode(connected=True)


class DataExploration(object):

    def __init__(self, dataframe):

        self.dataframe = dataframe

    def DesAnalysis(self, dis=False):
        """

            defining column name for feature statistics
            as the name suggested, we are capturing the following statistic from the features:
            Features: Name of Features
            Dtype: type of data
            Nunique: number of unique value
            freq1: most frequent value
            freq1_val: number of occurance of the most frequent value
            freq2: second most frequent value
            freq2_val: number of occurance of the second most frequent value
            freq3: 3rd most frequent value, if available
            freq3_val: number of occurance of the thrid most frequent value, if available
            describe stats: the following ones are the stat offer by our best friend .describe methods.

        """
        cols = self.dataframe.columns

        stat_cols= ['Dtype', 'Nunique', 'nduplicate', 'freq1', 'freq1_val', 'freq2', 'freq2_val',
             'freq3', 'freq3_val'] + self.dataframe[cols[0]].describe().index.tolist()[1:]

        stat_cols = ['Features']+stat_cols

        feature_stat = pd.DataFrame(columns=stat_cols)
        i = 0

        for col in cols:
            stat_vals = []

            # get stat value
            stat_vals.append(col)
            stat_vals.append(self.dataframe[col].dtype)
            stat_vals.append(self.dataframe[col].nunique())
            stat_vals.append(self.dataframe.shape[0]-self.dataframe[col].nunique())
            stat_vals.append(self.dataframe[col].value_counts().index[0])
            stat_vals.append(self.dataframe[col].value_counts().iloc[0])
            stat_vals.append(self.dataframe[col].value_counts().index[1])
            stat_vals.append(self.dataframe[col].value_counts().iloc[1])

            if len(self.dataframe[col].value_counts())>2:
                stat_vals.append(self.dataframe[col].value_counts().index[2])
                stat_vals.append(self.dataframe[col].value_counts().iloc[2])
            else:
                stat_vals.append(np.nan)
                stat_vals.append(np.nan)

            stat_vals += self.dataframe[col].describe().tolist()[1:]

            feature_stat.loc[i] = stat_vals
            i += 1




        # # number of unique
        # nunique_duplicate = self.dataframe.nunique().to_frame()
        # nunique_duplicate.columns = ['Nunique']
        # # number of duplcate
        # nunique_duplicate['Nduplicate'] = self.dataframe.shape[0]-nunique_duplicate['Nunique']
        # #         reset index
        # nunique_duplicate.reset_index(inplace=True)
        # nunique_duplicate = nunique_duplicate.rename(columns={'index': 'Features'})
        # # Type of data
        # nunique_duplicate['Dtype'] = pd.DataFrame([self.dataframe[f].dtype for f in self.dataframe.columns])

        # dipslay dataframe
        if dis:
            # display(nunique_duplicate)
            display(feature_stat)

        return feature_stat #nunique_duplicate

    def Describefloat(self):
        """
        Describe our categorical variable

        """
        pass
        # call_uni_dup = self.DesAnalysis()
        # float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
        # # display(self.dataframe[[w for w in float_exp]]) #.describe())
        # if dis:
        #     display(float_exp.loc[binary_exp])
        #
        # return float_exp.loc[binary_exp]

    def Describecat(self, cat=2, dis=False):
        """
        Describe our categorical variable

        """
        # pass
        call_uni_dup = self.DesAnalysis()
        binary_exp = call_uni_dup[call_uni_dup.Nunique == cat].index.values# .Features
        # [print(w) for w in binary_exp]
        # display(self.dataframe[[w for w in binary_exp]])#.describe())
        if dis:
            display(call_uni_dup.loc[binary_exp])
        # print(call_uni_dup)

        return call_uni_dup.loc[binary_exp]

    def PlotCat(self, cat=2):
        """
        Plot categorical vaiable
        cat: number of categorical variables in out dataframe

        """
        call_uni_dup = self.DesAnalysis()
        binary_exp = call_uni_dup[call_uni_dup.Nunique == cat].Features
        #
        for w in binary_exp:
            fig, ax = plt.subplots(2, 1, figsize=(20, 14))
            plt.subplot(211)
            plt.xlabel('count', fontsize=18)
            plt.ylabel(w, fontsize=18)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            sns.countplot(self.dataframe[w], palette='rainbow')
            # plt.show()
            plt.subplot(212)

            labels = []
            sizes = []
            for key, value in collections.Counter(self.dataframe[w]).items():
                labels.append(key)
                sizes.append(value)
            # Plot
            plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140)
            plt.axis('equal')

            plt.show()

    def PlotBinary(self, dataframe, bin_col):
        zero_list = []
        one_list = []
        for col in bin_col:
            zero_list.append((dataframe[col] == 0).sum())
            one_list.append((dataframe[col] == 1).sum())

        trace1 = go.Bar(
            x=bin_col,
            y=zero_list,
            name='Zero count'
        )
        trace2 = go.Bar(
            x=bin_col,
            y=one_list,
            name='One count'
        )

        data = [trace1, trace2]
        layout = go.Layout(
            barmode='stack',
            title='Count of 1 and 0 in binary variables'
        )

        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig, filename='stacked-bar')

    def PlotNumericalBar(self, type1='auto', type2="doane"):
        """
        Plot categorical vaiable
        cat: number of categorical variables in out dataframe
        'auto'
        'fd'
        'scott'
        'rice'
        'struges'
        'doane'
        'sqrt'
        """

        call_uni_dup = self.DesAnalysis()
        float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
        # #
        # # print(binary_exp)
        for w in float_exp:
            fig, ax = plt.subplots(1, 2, figsize=(15, 9))
            z = self.dataframe[w]
            plt.subplot(121)
            plt.hist(z, bins=type1)
            plt.title("{0} with {1} bins".format(w, type1), fontsize=20)
            plt.xlabel('xlabel', fontsize=18)
            plt.ylabel('ylabel', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.subplot(122)
            plt.hist(z, bins=type2)
            plt.title("{0} with {1} bins".format(w, type2), fontsize=20)
            plt.xlabel('xlabel', fontsize=18)
            plt.ylabel('count', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.show()
        #     print(self.dataframe[w].unique())
        #     print(self.dataframe[w].value_counts())

    def PlotNumericalBox(self):
        call_uni_dup = self.DesAnalysis()
        float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
        for w in float_exp:
            sns.boxplot(x=self.dataframe[w])
            plt.xlabel(w, fontsize=18)
            # plt.ylabel('count', fontsize=18)
            plt.xticks(fontsize=18)
            # plt.yticks(fontsize=18)
            plt.show()

    def DescribeFloatSkewKurt(self):
        """
            http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
            A fundamental task in many statistical analyses is to characterize
            the location and variability of a data set. A further
            characterization of the data includes skewness and kurtosis.
            Skewness is a measure of symmetry, or more precisely, the lack
            of symmetry. A distribution, or data set, is symmetric if it
            looks the same to the left and right of the center point.

            Kurtosis is a measure of whether the data are heavy-tailed
            or light-tailed relative to a normal distribution. That is,
            data sets with high kurtosis tend to have heavy tails, or
            outliers. Data sets with low kurtosis tend to have light
            tails, or lack of outliers. A uniform distribution would
            be the extreme case
        """
        call_uni_dup = self.DesAnalysis()
        float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
        for w in float_exp:
            print("{0} mean : ".format(w), np.mean(self.dataframe[w]))
            print("{0} var  : ".format(w), np.var(self.dataframe[w]))
            print("{0} skew : ".format(w), skew(self.dataframe[w]))
            print("{0} kurt : ".format(w), kurtosis(self.dataframe[w]))
            print('-*-'*25)

    def DealSkew(self, t='log'):
        """
            Input : sqrt, log, boxcox
            http://www.itl.nist.gov/div898/handbook/eda/section3/eda3.htm
            Many classical statistical tests and intervals depend on
            normality assumptions. Significant skewness and kurtosis
            clearly indicate that data are not normal. If a data set
            exhibits significant skewness or kurtosis (as indicated
            by a histogram or the numerical measures), what can we do about it?

            One approach is to apply some type of transformation
            to try to make the data normal, or more nearly normal.
            The Box-Cox transformation is a useful technique for
            trying to normalize a data set. In particular, taking
            the log or square root of a data set is often useful
            for data that exhibit moderate right skewness.

            Another approach is to use techniques based on
            distributions other than the normal. For example,
            in reliability studies, the exponential, Weibull,
            and lognormal distributions are typically used as
            a basis for modeling rather than using the normal
            distribution. The probability plot correlation
            coefficient plot and the probability plot are
            useful tools for determining a good distributional
            model for the data.
        """

        if t == 'log':
            call_uni_dup = self.DesAnalysis()
            float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
            for w in float_exp:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                x = self.dataframe[w]
                prob = stats.probplot(x, dist=stats.norm, plot=ax1)
                ax1.set_xlabel('')
                ax1.set_title('{0} Probplot against normal distribution'.format(w))
                ax2 = fig.add_subplot(212)
                xt = np.log(np.array(x.tolist())+1.)
                prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
                ax2.set_title('{0} Probplot after Log transformation'.format(w))
                plt.tight_layout()
                plt.show()

        elif t == 'sqrt':
            call_uni_dup = self.DesAnalysis()
            float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
            for w in float_exp:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                x = self.dataframe[w]
                prob = stats.probplot(x, dist=stats.norm, plot=ax1)
                ax1.set_xlabel('')
                ax1.set_title('{0} Probplot against normal distribution'.format(w))
                ax2 = fig.add_subplot(212)
                xt = np.sqrt(np.array(x.tolist())+1.)
                prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
                ax2.set_title('{0} Probplot after Log transformation'.format(w))
                plt.tight_layout()
                plt.show()

        elif t == 'boxcox':
            call_uni_dup = self.DesAnalysis()
            float_exp = call_uni_dup[call_uni_dup.Dtype == 'float'].Features
            for w in float_exp:

                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                x = self.dataframe[w] + max(z)
                prob = stats.probplot(x, dist=stats.norm, plot=ax1)
                ax1.set_xlabel('')
                ax1.set_title('{0} Probplot against normal distribution'.format(w))
                ax2 = fig.add_subplot(212)
                xt, _ = stats.boxcox(x)
                prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
                ax2.set_title('{0} Probplot after Box-Cox transformation'.format(w))
                plt.tight_layout()
                plt.show()
