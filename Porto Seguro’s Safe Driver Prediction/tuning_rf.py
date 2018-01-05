"""
Tuning Random Forest
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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



n_est = [80, 100, 150, 200]
n_dept = [5, 10]


name = ['rf0', 'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7']

l_score = {}
i = 0

for tune_n_est in n_est:
    print()
    print('//\\'*13)
    print('Tuning Random Forest = {0}'.format(tune_n_est))

    for tune_n_dept in n_dept:
        print()
        print('//'*25)
        print('Tuning Random Forest n_dept = {0}'.format(tune_n_dept))

        rf1 = RandomForestClassifier(n_estimators=tune_n_est,
                                     n_jobs=6,
                                     min_samples_split=5,
                                     max_depth=tune_n_dept,
                                     criterion='gini',
                                     random_state=0)

        l_base_models = [(name[i], [rf1, False, True])]

        l_testp = ML.Ensemble(nnew_train,
                         'target',
                         ['id', 'target'],
                         nnew_test,
                         'id',
                         l_base_models,
                         5)

        l_lv1_score, l_lv1_train, l_lv1_test = l_testp.FitModel()
        l_score.update(l_lv1_score)
        i = i + 1

print(pd.DataFrame([l_score]))
