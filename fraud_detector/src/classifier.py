from fraud import FraudData, TransactionData, DataSplitter
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas
from xgboost import XGBClassifier
import os
from catboost import CatBoostClassifier, Pool
import numpy
from sklearn.model_selection import KFold
from collections import defaultdict
from feature_extractor import FraudFeatureExtractor


class KaggleSubmitMixin:
    """
    Mixin class for all estimators to have submit file in correct format

    """
    def submit(self, xtest=None, trans_ids=None):
        """
        Saves prediction outputs in kaggle submission file format

        :param xtest: competition test data
        :param trans_ids: transaction ids
        :return: saved kaggle file format

        """
        test_predictions = self.predict(xtest)[:, 1]
        my_submission_file = pandas.DataFrame()
        my_submission_file['TransactionID'] = trans_ids
        my_submission_file['isFraud'] = test_predictions
        my_submission_file.to_csv('submission.csv', index=False)


class BaggedRFModel(BaseEstimator, KaggleSubmitMixin):
    """
    Bagging model with Randomforest as base estimator
    """
    rf_params = {'n_estimators': 500,
                 'max_depth': 9,
                 'n_jobs': -1}

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.fraud_transformer = FraudFeatureExtractor()
        self.classifier = None

    def fit(self, xtrain=None, ytrain=None):
        """
        Fit bagging model with Random Forest based estimator

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :return: BaggedRFModel objects
        """
        self.fraud_transformer.fit(xtrain, ytrain)
        xtrain = self.fraud_transformer.transform(xtrain)
        estimator = RandomForestClassifier(**self.rf_params)

        self.classifier = BaggingClassifier(base_estimator=estimator, n_estimators=self.n_estimators)
        self.classifier.fit(xtrain, ytrain)
        return self

    def predict(self, xtest=None):
        """
        Apply trained classifier on test data

        :param xtest: pandas data_frame for test data
        :return: array of predicted probability values

        """
        xtest = self.fraud_transformer.transform(xtest)
        return self.classifier.predict_proba(xtest)


class CatBoostModel(BaseEstimator, KaggleSubmitMixin):
    """
    Estimator using Catboost
    """
    _SEED = 1

    def __init__(self):
        self.fraud_transformer = FraudFeatureExtractor()
        self.classifier = None

    def fit(self, xtrain=None, ytrain=None, xval=None, yval=None):
        """
        Fit Catboost model

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: CatBoostModel object
        """
        # If no validation data is specified, use training data
        if not any([xval, yval]):
            xval = xtrain
            yval = ytrain

        self.fraud_transformer.fit(xtrain, ytrain)
        xtrain = self.fraud_transformer.transform(xtrain)
        xval = self.fraud_transformer.transform(xval)

        nr_cat_features = int(numpy.sum(xtrain.max(axis=0) == 1))
        cat_features = list(range(nr_cat_features))
        train_data = Pool(data=xtrain,
                          label=ytrain,
                          cat_features=cat_features)
        valid_data = Pool(data=xval,
                          label=yval,
                          cat_features=cat_features)
        params = {'loss_function': 'Logloss',
                  'eval_metric': 'AUC',
                  'cat_features': cat_features,
                  'iterations': 2000,
                  'verbose': 10,
                  'max_depth': 7,
                  'random_seed': self._SEED,
                  'od_type': "Iter",
                  'od_wait': 100,
                  }
        self.classifier = CatBoostClassifier(**params)
        self.classifier.fit(train_data, eval_set=valid_data)
        return self

    def predict(self, xtest=None):
        """
        Apply trained classifier on test data

        :param xtest:  pandas data_frame for test data
        :return: array of predicted probability values
        """
        xtest = self.fraud_transformer.transform(xtest)
        return self.classifier.predict_proba(xtest)


class XGBModel(BaseEstimator, KaggleSubmitMixin):
    def __init__(self):
        self.fraud_transformer = FraudFeatureExtractor()
        self.classifier = None

    def fit(self, xtrain=None, ytrain=None, xval=None, yval=None):
        """
        Fit XGM based model

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: XGBModel object
        """
        # If no validation data is specified, use training data
        if not any([xval, yval]):
            xval = xtrain.copy()
            yval = ytrain.copy()

        xtrain = self.fraud_transformer.fit_transform(xtrain, ytrain)
        xval = self.fraud_transformer.transform(xval)
        eval_set = [(xval, yval)]
        self.classifier = XGBClassifier(n_estimators=2000,
                                        max_depth=9,
                                        learning_rate=0.048,
                                        subsample=0.9,
                                        colsample_bytree=0.9,
                                        reg_alpha=0.5,
                                        reg_lamdba=0.5,
                                        n_jobs=-1)
        self.classifier.fit(xtrain, ytrain, eval_metric=["error", "logloss"],
                            early_stopping_rounds=100, eval_set=eval_set, verbose=True)

    def predict(self, xtest=None):
        """
        Apply trained classifier on test data

        :param xtest: pandas data_frame for test data
        :return: array of predicted probability values
        """
        xtest = self.fraud_transformer.transform(xtest)
        return self.classifier.predict_proba(xtest)


class KFoldModel(BaseEstimator, KaggleSubmitMixin):
    """
    Apply models on different fold splits of the data

    """
    _SEED = 1
    _SPLITS = 5

    def __init__(self, n_estimators=500, split_feature='card6'):
        self.n_estimators = n_estimators
        self.classifier = None
        self.split_estimators = defaultdict(list)
        self.split_transformers = defaultdict(list)
        self.split_feature = split_feature
        self.fraud_transformer = FraudFeatureExtractor()
        self.estimators = []

    def fit(self, xtrain=None, ytrain=None):
        """
        Fits _SPLITS number of models

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: KFoldModel object
        """
        self.fraud_transformer.fit(xtrain, ytrain)
        xtrain = self.fraud_transformer.transform(xtrain, ytrain)

        folds = KFold(n_splits=self._SPLITS, shuffle=True)
        for fold_n, (train_index, valid_index) in enumerate(folds.split(xtrain)):
            self.classifier = XGBClassifier(n_estimators=10,
                                            max_depth=9,
                                            learning_rate=0.048,
                                            subsample=0.85,
                                            colsample_bytree=0.85,
                                            reg_alpha=0.15,
                                            reg_lamdba=0.85,
                                            n_jobs=-1)
            x_train_, x_valid = xtrain[train_index, :], xtrain[valid_index, :]
            y_train_, y_valid = ytrain[train_index], ytrain[valid_index]
            eval_set = [(x_valid, y_valid)]
            fit_params = {'eval_metric': ["error", "logloss"],
                          'early_stopping_rounds': 100,
                          'eval_set': eval_set,
                          'verbose': True}
            self.classifier.fit(x_train_, y_train_, **fit_params)
            self.estimators.append(self.classifier)
        return self

    def predict(self, xtest=None):
        """
        Apply trained classifier on test data

        :param xtest: Test data
        :return: array of predicted probability values

        """
        xtest = self.fraud_transformer.transform(xtest)
        pred = numpy.zeros((xtest.shape[0], 2))
        for clf in self.estimators:
            pred += clf.predict_proba(xtest)/self._SPLITS
        return pred


class ClusteredXGBModel(BaseEstimator, KaggleSubmitMixin):
    """
    Splits data on group and apply model on each group

    """
    _SEED = 1
    _SPLITS = 5

    def __init__(self, split_feature='card6'):
        self.split_estimators = defaultdict(list)
        self.split_transformers = defaultdict(list)
        self.split_feature = split_feature
        self.fraud_transformer = FraudFeatureExtractor()
        self.classifier = None

    def fit(self, xtrain=None, ytrain=None):
        """
        Combination of cluster classifier and bagging on samples

        :param xtrain: pandas data_frame containing training data
        :param ytrain: numpy array with model targets
        :return: ClusteredXGBModel object
        """
        xtrain['target'] = ytrain

        for name, group in xtrain.groupby(self.split_feature):
            group_target = group['target'].values
            group_features = group.drop('target', axis=1)
            group_transformer = FraudFeatureExtractor()
            group_transformer.fit(group_features, group_target)
            group_features = group_transformer.transform(group_features)

            folds = KFold(n_splits=self._SPLITS, shuffle=True)
            for fold_n, (train_index, valid_index) in enumerate(folds.split(group_features)):
                x_train_, x_valid = group_features[train_index, :], group_features[valid_index, :]
                y_train_, y_valid = group_target[train_index], group_target[valid_index]
                eval_set = [(x_valid, y_valid)]
                estimator = XGBClassifier(n_estimators=1500,
                                          max_depth=9,
                                          learning_rate=0.048,
                                          subsample=0.85,
                                          colsample_bytree=0.85,
                                          reg_alpha=0.15,
                                          reg_lamdba=0.85,
                                          n_jobs=-1)
                estimator.fit(x_train_, y_train_, eval_metric=["error", "logloss"],
                              early_stopping_rounds=100, eval_set=eval_set, verbose=True)
                self.split_estimators[name].append(estimator)
                self.split_transformers[name].append(group_transformer)
        return self

    def predict(self, xtest=None):
        """
        Transform and Predict based on splits of the split feature

        :param xtest: pandas dataframe with test data
        :return: model probabilities
        """
        predictions = numpy.zeros((xtest.shape[0], 2))
        for name, group in xtest.groupby(self.split_feature):
            for s in range(self._SPLITS):
                group_transformed = self.split_transformers[name][s].transform(group)
                predictions[group.index.values, :] += self.split_estimators[name][s].predict_proba(group_transformed) / self._SPLITS
        return predictions


if __name__ == '__main__':

    if os.environ.get('USER', 'notflash') == 'flash':
        data = DataSplitter(transaction_file='train_transaction_5000.csv', identity_file='train_identity_5000.csv')
    else:
        data = DataSplitter(transaction_file='train_transaction.csv', identity_file='train_identity.csv')

    # fraud_transformer = FraudFeatureExtractor()
    # m = transformer.mapper.transformer_list[0][1][1]
    # fraud_transformer.fit(data.X_train)
    X_train = data.X_train
    X_test = data.X_test

    print('X_train shape: ', X_train.shape)
    # apply classifier
    model = FraudDetector()

    model.fit(X_train, data.y_train)
    # X_test = data.X_test # fraud_transformer.transform(data.X_test)

    y_predictions = model.predict(X_test)[:, 1]
    print('AUC on Train set: ', roc_auc_score(data.y_train, model.predict(X_train)[:, 1]))
    print('AUC on Test set: ', roc_auc_score(data.y_test, y_predictions))

    # Apply model on test results
    if os.environ.get('USER', 'notflash') != 'flash':
        fraud = FraudData(transaction_file='test_transaction.csv',
                          identity_file='test_identity.csv')
        # fraud = FraudData(transaction_file='train_transaction_5000.csv',
        #                   identity_file='train_identity_5000.csv')
        # fraud = TransactionData(file_name='test_transaction.csv')
        model.submit(fraud.data, trans_ids=fraud.data['TransactionID'])

# encoder_steps = fraud_transformer.mapper.transformer_list[0][1][1]
# nr_cat_features = len(encoder_steps.keys())

