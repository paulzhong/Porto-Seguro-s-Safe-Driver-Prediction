import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class FeaturesSelection(object):
    """
      feature selection by random forest
    """
    def DefineXTrainYtrain(self, train, train_col_drop, test_col_drop):
        """
        return sliced train and data set
            input:
                train
                train_col_drop: drop ['id','target'] from train
                test_col_drop: chose 'target'

            return:
                X_train: drop IDs and Target from train
                y_train: chose Target column
        """
        X_train = train.drop(train_col_drop, axis=1)
        y_train = train[test_col_drop]
        return X_train, y_train

    def RFFeatureImp(self, X_train, y_train):
        """
            random forest feature importance
            input:
                X_train
                y_train
            return:
                rf
                feat_labels
        """
        feat_labels = X_train.columns

        rf = RandomForestClassifier(n_estimators=1000,
                                    random_state=0,
                                    n_jobs=-1)
        # fitting
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_

        indices = np.argsort(rf.feature_importances_)[::-1]

        for f in range(X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]]))

        return rf, feat_labels

    def SelectingFeatures(self, X_train, rf, feat_labels):

        sfm = SelectFromModel(rf, threshold='median', prefit=True)
        print('Number of features before selection: {}'.format(X_train.shape[1]))
        n_features = sfm.transform(X_train).shape[1]
        print('Number of features after selection: {}'.format(n_features))
        selected_vars = list(feat_labels[sfm.get_support()])

        train_selected_vars = ['id', 'target'] + selected_vars
        test_selected_vars = ['id'] + selected_vars

        return train_selected_vars, test_selected_vars

    def MainFeatureSelection(self, train, train_col_drop, test, test_col_drop):
        """
        """
        assert type(train_col_drop) == list, "columns is not list "
        for w in train_col_drop: assert w in train.columns, 'Column name is not in train'
        assert test_col_drop in train.columns, 'Column name is not in test'

        X_train, y_train = self.DefineXTrainYtrain(train,
                                                   train_col_drop,
                                                   test_col_drop)

        rf, feat_labels = self.RFFeatureImp(X_train, y_train)

        train_selected_vars, test_selected_vars = self.SelectingFeatures(X_train, rf, feat_labels)

        return train[train_selected_vars], test[test_selected_vars]
