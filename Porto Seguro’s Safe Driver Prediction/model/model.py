import numpy as np
import pandas as pd

from crossValidation.cvsklearn import cross_validate_sklearn
from crossValidation.cvxgb import cross_validate_xgb
from crossValidation.cvlgb import cross_validate_lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
import time

from sklearn.model_selection import StratifiedKFold


class Ensemble(object):
    """
        Stacking models
        input:
            train
            target_col(str): the name of the target column
            train_drop_col(list of str): ['id','target'], the name of columns which we want to drop from train
            test
            test_drop_col(str): 'id', the name of columns which we want to drop from test
            base_models = [('rf',model, scale=True/False, True/False),
                           ('xgb',model2, use_rank=True/False, verbose_eval=True/False)
                           ('lgb',model1, use_rank=True/False, verbose_eval=True/False, use_cat=True/False)]

        output:
                train and test dataframes of the prediction

    """



    def __init__(self, train, target_col,
                 train_drop_col,
                 test, test_drop_col,
                 base_models,
                 n_splits):

        self.n_splits = n_splits
        self.base_models = base_models
#********************************************** orginal 2017
        self.kf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2017)

        if type(target_col)==str:
            self.y_train = train[target_col]
        else:
            self.y_train = target_col

        if train_drop_col:
            self.x_train = train.drop(train_drop_col, axis=1)
        else:
            self.x_train = train

        if test_drop_col:
            self.x_test = test.drop([test_drop_col], axis=1)
        else:
            self.x_test = test




        assert self.x_train.shape[1] == self.x_test.shape[1], 'Number of columns of train and test do not match'



    """
        rf: Random Forest
        logit: LogisticRegression
        nb: BernoulliNB
        xgb: XGBoost
        lgb: LightGBM

    """

    def FitModel(self):
        """
        """
        machine_learning = {'sklearn_machine_learning' : ['rf'+str(i) for i in range(30)] +
                                                         ['nb'+str(i) for i in range(30)] +
                                                         ['logit'+str(i) for i in range(30)],
                            'XGBoost':['xgb'+str(i) for i in range(30)],
                            'LightGBM':['lgb'+str(i) for i in range(30)]}

        model_dic = dict(self.base_models)
        #

        dict_of_models_cv, dict_of_models_train, dict_of_models_test = {}, {}, {}

        #
        i, j, k = 0, 0, 0

        for key,value in model_dic.items():

            if key in machine_learning['sklearn_machine_learning']:
                print()
                print('-*- '*20)
                print('sklearn: ' + key)
                print('-'*20)

                outcomes = cross_validate_sklearn(value[0],
                                                 self.x_train, self.y_train, self.x_test, self.kf,
                                                 scale=value[1], verbose=value[2])

                dict_of_models_cv[key] = outcomes[0]
                dict_of_models_train[key] = outcomes[1]
                dict_of_models_test[key] = outcomes[2]

                i += 1

            if key in machine_learning['XGBoost']:
                print()
                print('-*- '*20)
                print('xgb: ' + key)
                print('-'*20)

                outcomes = cross_validate_xgb(value[0],
                                              self.x_train, self.y_train, self.x_test, self.kf,
                                              use_rank=value[1], verbose_eval=value[2])

                dict_of_models_cv[key] = outcomes[0]
                dict_of_models_train[key] = outcomes[1]
                dict_of_models_test[key] = outcomes[2]

                j += 1

            if key in machine_learning['LightGBM']:
                print()
                print('-*- '*20)
                print('lgb: ' + key )
                print('-'*20)

                outcomes = cross_validate_lgb(value[0],
                                              self.x_train, self.y_train, self.x_test, self.kf,
                                              verbose_eval=value[2], use_rank=value[1],use_cat=value[3])


                dict_of_models_cv[key] = outcomes[0]
                dict_of_models_train[key] = outcomes[1]
                dict_of_models_test[key] = outcomes[2]

                k +=1


        return dict_of_models_cv, pd.DataFrame(dict_of_models_train),pd.DataFrame(dict_of_models_test)
