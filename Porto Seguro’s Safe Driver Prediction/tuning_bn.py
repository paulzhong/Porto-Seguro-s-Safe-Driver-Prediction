"""
Tuning Random Forest
"""

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import model.model as ML


def LoadingData(path_to_train, path_to_test):
    try:
        train = pd.read_csv(path_to_train)
        test = pd.read_csv(path_to_test)
        return train, test
    except FileNotFoundError:
        print("The path does not exist")


nnew_train, nnew_test = LoadingData('./data/nnew_train.csv',
                                    './data/nnew_test.csv')
nnew_train = nnew_train.drop(['Unnamed: 0'], axis=1)
nnew_test = nnew_test.drop(['Unnamed: 0'], axis=1)


tt = [0.7, 0.8, 0.9, 1.0]
name = ['nb0', 'nb1', 'nb3', 'nb4']
l_score = {}
i = 0
for s in tt:
    print()
    print('//\\'*13)
    print('Tuning BN alpha = {0}'.format(s))
    nb = BernoulliNB(alpha=s)


    l_base_models = [(name[i], [nb, True, True])]

    l_testp = ML.Ensemble(nnew_train,
                     'target',
                     ['id', 'target'],
                     nnew_test,
                     'id',
                     l_base_models,
                     5)

    l_lv1_score, l_lv1_train, l_lv1_test = l_testp.FitModel()
    l_score.update(l_lv1_score)
    i += 1


nb_score = l_score
print(pd.DataFrame([nb_score]))
