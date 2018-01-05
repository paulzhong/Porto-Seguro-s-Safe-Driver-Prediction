import pandas as pd
from crossValidation.cvxgb import cross_validate_xgb
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

t_max_depth = [3, 4, 5]
t_gamma = [0.6, 0.7, 0.9]


name = ['xgb0', 'xgb1', 'xgb2', 'xgb3',
        'xgb4', 'xgb5', 'xgb6', 'xgb7', 'xgb8']


l_score = {}
i = 0

for tune_max_depth in t_max_depth:
    print()
    print('//\\'*13)
    print('Tuning xgb max_depth = {0}'.format(tune_max_depth))
    for tune_gamma in t_gamma:
        print()
        print('//'*25)
        print('Tuning xgb gamma = {0}'.format(tune_gamma))
        # XGBoost
        xgb_params = {
          "booster"  :  "gbtree",
          "objective"         :  "binary:logistic",
          "tree_method": "hist",
          "eval_metric": "auc",
          "eta": 0.1,
          "max_depth": tune_max_depth,
          "min_child_weight": 10,
          "gamma": tune_gamma,
          "subsample": 0.76,

          "colsample_bytree": 0.95,
          "nthread": 6,
          "seed": 0,
          'silent': 1}

        base_models = [(name[i], [xgb_params, False, False])]

        l_testp = ML.Ensemble(nnew_train,
                       'target',
                       ['id', 'target'],
                       nnew_test,
                       'id',
                       base_models,
                       5)

        l_lv1_score, l_lv1_train, l_lv1_test = l_testp.FitModel()
        l_score.update(l_lv1_score)
        i = i + 1


print(pd.DataFrame([l_score]))
