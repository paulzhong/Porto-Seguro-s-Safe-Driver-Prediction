import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time

from crossValidation.gini import auc_to_gini_norm
from crossValidation.probabilityToRank import probability_to_rank


def cross_validate_lgb(params, x_train, y_train, x_test, kf, cat_cols=[],
                       verbose=True, verbose_eval=50, use_cat=True, use_rank=True):
    start_time = time.time()
    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    if len(cat_cols) == 0: use_cat = False

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        if use_cat:
            lgb_train = lgb.Dataset(x_train_kf, y_train_kf, categorical_feature=cat_cols)
            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train, categorical_feature=cat_cols)
        else:
            lgb_train = lgb.Dataset(x_train_kf, y_train_kf)
            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=4000,
                        valid_sets=lgb_val,
                        early_stopping_rounds=30,
                        verbose_eval=verbose_eval)

        val_pred = gbm.predict(x_val_kf)

        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred += probability_to_rank(gbm.predict(x_test))
            # test_pred += gbm.predict(x_test)
        else:
            train_pred[val_index] += val_pred
            test_pred += gbm.predict(x_test)

        # test_pred += gbm.predict(x_test)
        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)
        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))
    return cv_score, train_pred, test_pred
