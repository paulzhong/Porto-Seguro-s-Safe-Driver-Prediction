import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time

from crossValidation.gini import auc_to_gini_norm
from crossValidation.probabilityToRank import probability_to_rank

"""
Xgboost K-fold & OOF function

"""


# def probability_to_rank(prediction, scaler=1):
#     """
#     convert probability into rank for these two OOF function.
#     The needs to use normalised rank instead of predicted probabilities will
#     become appearent later in this notebook
#     """
#     pred_df = pd.DataFrame(columns=['probability'])
#     pred_df['probability'] = prediction
#     pred_df['rank'] = pred_df['probability'].rank()/len(prediction)*scaler
#     return pred_df['rank'].values


def cross_validate_xgb(params, x_train, y_train, x_test,
                       kf, cat_cols=[], verbose=True,
                       verbose_eval=50, num_boost_round=4000,
                       use_rank=True):
    """
    the k-fold function for XGB to generate OOF predictions, this function is
    very much similar to its sklearn counter part. The difference is that
    we need to use the XGB interface to facilitate the classifer,
    also we provide an option cover probability into rank
    """
    start_time = time.time()

    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        x_test_kf = x_test.copy()

        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)
        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)
        d_test = xgb.DMatrix(x_test_kf)

        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,
                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,
                        early_stopping_rounds=50)

        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)
        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred += probability_to_rank(bst.predict(d_test))
        else:
            train_pred[val_index] += val_pred
            test_pred += bst.predict(d_test)

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc,
                                                                                     fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

        return cv_score, train_pred, test_pred
