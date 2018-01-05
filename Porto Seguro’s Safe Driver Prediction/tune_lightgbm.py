import pandas as pd
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

t_learning_rate = [0.05, 0.1, 0.5]
t_max_depth = [4, 5, 6]

l_score = {}
i = 0

for tune_max_depth in t_max_depth:
    print()
    print('//\\'*13)
    print('Tuning lgb max_depth = {0}'.format(tune_max_depth))
    for tune_learning in t_learning_rate:
        print()
        print('//'*25)
        print('Tuning xgb gamma = {0}'.format(tune_learning))
        # LightGBoost
        lgb_params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 22,
            'min_sum_hessian_in_leaf': 20,
            'max_depth': tune_max_depth,
            'learning_rate': tune_learning,#0.1,  # 0.618580
            'num_threads': 6,
            'feature_fraction': 0.6894,
            'bagging_fraction': 0.4218,
            'max_drop': 5,
            'drop_rate': 0.0123,
            'min_data_in_leaf': 10,
            'bagging_freq': 1,
            'lambda_l1': 1,
            'lambda_l2': 0.01,
            'verbose': 1
        }

        base_models = [('lgb0', [lgb_params, False, False, True])]

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
