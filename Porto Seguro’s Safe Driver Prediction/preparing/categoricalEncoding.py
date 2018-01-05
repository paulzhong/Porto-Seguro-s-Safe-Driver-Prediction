import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures


class FeaturesEng(object):

    def FreqEncoding(self, cols, train_df, test_df):
        """
            This function late in a list of features 'cols' from train and test dataset,
            and performing frequency encoding.

            arguments:
            cols: name of columns whoch is categorical
            train_df: train dataset
            test_df: test dataset

            return:

         """
        # we are going to store our new dataset in these two resulting datasets
        result_train_df = pd.DataFrame()
        result_test_df = pd.DataFrame()

        # loop through each feature column to do this
        for col in cols:

            # capture the frequency of a feature in the training set in the form of a dataframe
            col_freq = col + '_freq'
            freq = train_df[col].value_counts()
            freq = pd.DataFrame(freq)
            freq.reset_index(inplace=True)
            freq.columns = [[col, col_freq]]

            # merge ths 'freq' datafarme with the train data
            temp_train_df = pd.merge(train_df[[col]], freq, how='left', on=col)
            temp_train_df.drop([col], axis=1, inplace=True)

            # merge this 'freq' dataframe with the test data
            temp_test_df = pd.merge(test_df[[col]], freq, how='left', on=col)
            temp_test_df.drop([col], axis=1, inplace=True)

            # if certain levels in the test dataset is not observed in the train dataset,
            # we assign frequency of zero to them
            temp_test_df.fillna(0, inplace=True)
            temp_test_df[col_freq] = temp_test_df[col_freq].astype(np.int32)

            if result_train_df.shape[0] == 0:
                result_train_df = temp_train_df
                result_test_df = temp_test_df
            else:
                result_train_df = pd.concat([result_train_df, temp_train_df], axis=1)
                result_test_df = pd.concat([result_test_df, temp_test_df], axis=1)

        return result_train_df, result_test_df

    def BinaryEncoding(self, train_df, test_df, feat):
        """
            perform binary encoding for categorical variable
            this function take in a pair of train and test data set, and the feature that need to be encode.
            it returns the two dataset with input feature encoded in binary representation
            this function assumpt that the feature to be encoded is already been encoded in a numeric manner
            ranging from 0 to n-1 (n = number of levels in the feature)
        """
        # calculate the highest numerical value used for numeric encoding
        train_feat_max = train_df[feat].max()
        test_feat_max = test_df[feat].max()
        if train_feat_max > test_feat_max:
            feat_max = train_feat_max
        else:
            feat_max = test_feat_max

        # use the value of feat_max+1 to represent missing value
        train_df.loc[train_df[feat] == -1, feat] = feat_max + 1
        test_df.loc[test_df[feat] == -1, feat] = feat_max + 1

        # create a union set of all possible values of the feature
        union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())

        # extract the highest value from from the feature in decimal format.
        max_dec = union_val.max()

        # work out how the ammount of digtis required to be represent max_dev in binary representation
        max_bin_len = len("{0:b}".format(max_dec))
        index = np.arange(len(union_val))
        columns = list([feat])

        # create a binary encoding feature dataframe to capture all the levels for the feature
        bin_df = pd.DataFrame(index=index, columns=columns)
        bin_df[feat] = union_val

        # capture the binary representation for each level of the feature
        feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))

        # split the binary representation into different bit of digits
        splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
        splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]
        bin_df = bin_df.join(splitted)

        # merge the binary feature encoding dataframe with the train and test dataset - Done!
        train_df = pd.merge(train_df, bin_df, how='left', on=[feat])
        test_df = pd.merge(test_df, bin_df, how='left', on=[feat])
        return train_df, test_df

    def add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def Smoothing(self,
                      trn_series=None,
                      tst_series=None,
                      target=None,
                      min_samples_leaf=1,
                      smoothing=1,
                      noise_level=0):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior
        """
        assert len(trn_series) == len(target)
        assert trn_series.name == tst_series.name
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, noise_level), self.add_noise(ft_tst_series, noise_level)

    def Concat(self, dataframe1, dataframe2):

        # Concatinate two dataframes
        return pd.concat([dataframe1, dataframe2], axis=1)

    def Interaction(self, x, n_col):
        """
        """
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        v = n_col.tolist()
        interactions = pd.DataFrame(data=poly.fit_transform(x[v]), columns=poly.get_feature_names(v))
        interactions.drop(v, axis=1, inplace=True)  # Remove the original columns
        # Concat the interaction variables to the data
        print('Before creating interactions we have {} variables'.format(x.shape[1]))
        x = pd.concat([x, interactions], axis=1)
        print('After creating interactions we have {} variables'.format(x.shape[1]))
        return x

    def NumericalInteractions(self, train, test, n_col):
        return self.Interaction(train, n_col), self.Interaction(test, n_col)
