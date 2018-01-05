from sklearn.utils import shuffle
from operator import itemgetter


class Sampling(object):
    """
    input: dataframe

    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def TargetCountUniqueDic(self, target):
        """
        make dic for target count for two classes 0,1
        input: name of target column and datafarme
        return dic of target and number of uniqe
        """
        dic = {self.dataframe[target].value_counts().index[0]:
               self.dataframe[target].value_counts().iloc[0],
               self.dataframe[target].value_counts().index[1]:
               self.dataframe[target].value_counts().iloc[1]}
        return dic

    def SortDicValue(self, mydic):
        """
        return sorted dic as a list of tubels
        """
        return sorted(mydic.items(), key=itemgetter(1))

    def UnderSampling(self, target, desired_apriori):

        """

        method: UnderSampling from bigger sample
        tagrget: 0 ,1
        input: desired_apriori

        retrun sample_train

        """
        value = self.SortDicValue(self.TargetCountUniqueDic(target))
        # Get the indices per target value
        idx_big = self.dataframe[self.dataframe[target] == value[1][0]].index
        idx_small = self.dataframe[self.dataframe[target] == value[0][0]].index

        # Get original number of records per target value
        nb_0 = len(self.dataframe.loc[idx_big])
        nb_1 = len(self.dataframe.loc[idx_small])

        # Calculate the undersampling rate and resulting number of records with target=0
        undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
        undersampled_nb_0 = int(undersampling_rate*nb_0)
        print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
        print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

        # Randomly select records with target=0 to get at the desired a priori
	# random_state=37 orginal... 2018
        undersampled_idx = shuffle(idx_big, random_state=37, n_samples=undersampled_nb_0)

        # Construct list with remaining indices
        idx_list = list(undersampled_idx) + list(idx_small)

        # Return undersample data frame
        sample_train = self.dataframe.loc[idx_list].reset_index(drop=True)
        return sample_train
