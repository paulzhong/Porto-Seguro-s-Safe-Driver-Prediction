"""
Tuning Logistic Regression
"""
# import numpy as np
import pandas as pd
# from operator import itemgetter

# from crossValidation.cvsklearn import cross_validate_sklearn

# from sklearn.metrics import roc_auc_score

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold

import model.model as ML

def LoadingData(path_to_train, path_to_test):
    try:
        train = pd.read_csv(path_to_train)
        test = pd.read_csv(path_to_test)
        return train, test
    except FileNotFoundError:
        print("The path does not exist")


nnew_train, nnew_test = LoadingData('./data/nnew_train.csv', './data/nnew_test.csv')

nnew_train = nnew_train.drop(['Unnamed: 0'], axis=1)
nnew_test = nnew_test.drop(['Unnamed: 0'], axis=1)


# Logistic Regression tunining
penalt = ['l1', 'l2']
c = [0.1, 0.2, 0.5, 0.6]
#
name = ['logit0', 'logit1', 'logit2', 'logit3',
        'logit4', 'logit5', 'logit6', 'logit7']
l_score={}
i = 0
#
for tune_pen in penalt:
    print()
    print('//\\'*13)
    print('Tuning logistic regression penalty = {0}'.format(tune_pen))

    for tune_c in c:
        print()
        print('//'*25)
        print('Tuning logistic regression c = {0}'.format(tune_c))
        logit = LogisticRegression(penalty=tune_pen, random_state=0, C=tune_c)

        l_base_models = [(name[i], [logit, True, True])]

        l_testp = ML.Ensemble(nnew_train,
                         'target',
                         ['id','target'],
                         nnew_test,
                         'id',
                         l_base_models,
                         5)

        l_lv1_score, l_lv1_train, l_lv1_test = l_testp.FitModel()
        l_score.update(l_lv1_score)
        i = i + 1

print(pd.DataFrame([l_score]))
