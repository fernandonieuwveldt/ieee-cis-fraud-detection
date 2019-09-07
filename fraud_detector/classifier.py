from fraud_detector import INPUT_DIR
from fraud_detector import OUTPUT_DIR
from detector import FraudData
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from os import path
from constants import emails
import pandas
from detector import DataSplitter
import sklearn_pandas
import copy


numerical_features = ['card1', 'card2', 'card3', 'card5']
                      # 'addr1', 'addr2',
                      # 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that select columns based on type or from list of names
    """
    def __init__(self, type=None):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = X.dtypes[X.dtypes == self.type].index
        return X[features]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, xc, y=None):
        xc_copy = xc.copy()
        important_features = ['card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'ProductCD', 'DeviceType', 'DeviceInfo']
        x_snipped = xc_copy[important_features]
        x_snipped.loc[:, 'P_emaildomain'] = x_snipped.loc[:, 'P_emaildomain'].map(emails)
        x_snipped.loc[:, 'R_emaildomain'] = x_snipped.loc[:, 'R_emaildomain'].map(emails)
        return x_snipped.fillna('Unknown')


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, xn=None):
        xn_copy = xn.copy()
        # xn_copy.fillna('-999', inplace=True)
        object_types = xn_copy.dtypes[xn.dtypes == 'object'].index
        xn_copy.drop(object_types, axis=1, inplace=True)
        return xn_copy


class FraudFeatureExtractor(TransformerMixin):
    def __init__(self):
        self.mapper = None

    @staticmethod
    def fraud_detector_extractor_mapper():
        numerical_pipeline = Pipeline(steps=[('numerical_features', NumericalTransformer()),
                                             ('imputer', SimpleImputer(strategy='mean')),
                                             ('scaler', StandardScaler())
                                            ])

        categorical_pipeline = Pipeline(steps=[('categorical_features', CategoricalTransformer()),
                                               #('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(sparse=False))
                                               ])

        transformer_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                              ('numerical_pipeline', numerical_pipeline)])
        return transformer_pipeline

    def fit(self, samples):
        self.mapper = self.fraud_detector_extractor_mapper().fit(samples)
        return self

    def transform(self, x=None):
        return self.mapper.transform(x)


class FraudDetector(BaseEstimator):
    """
    Fraud detector model

    """

    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.classifier = None  # this will be another sklearn estimator

    def fit(self, xtrain=None, ytrain=None):
        """
        Fit model on FraudData

        :param xtrain: table with features
        :param ytrain: target variable
        :return:
        """
        self.classifier = Pipeline(steps=[#('clf', RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1))
                                          #('scaler', StandardScaler()),
                                          ('clf', RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=2))
                                          ])
        #dt = DecisionTreeClassifier(max_depth=5, random_state=1)
        #self.classifier = AdaBoostClassifier(base_estimator=dt, n_estimators=self.n_estimators)
        self.classifier.fit(xtrain, ytrain)
        return self

    def predict(self, xtest=None):
        """
        Apply trained classifier on test data

        :param xtest: Test data
        :return: array of predicted probability values

        """
        return self.classifier.predict_proba(xtest)

    def submit(self, xtest=None):
        """
        Saves prediction outputs in kaggle submission file format

        :param xtest: competition test data
        :return: saved kaggle file format

        """
        test_predictions = self.predict(xtest)[:, 1]
        my_submission_file = pandas.DataFrame()
        my_submission_file['TransactionID'] = xtest['TransactionID']
        my_submission_file['isFraud'] = test_predictions

        my_submission_file.to_csv(path.join(OUTPUT_DIR, 'submission.csv'))


if __name__ == '__main__':

    # fraud = FraudData(transaction_file='train_transaction.csv',
    #                   identity_file='train_identity.csv')
    # # cat_transformer = CategoricalTransformer()
    # # # filled_data = cat_transformer.fit_transform(fraud.data)
    # object_pipeline = Pipeline(steps=[('object_selector', TypeSelector('object')),
    #                                   ('object_transformer', CategoricalTransformer()),
    #                                   ('label_encoder', LabelEncoder())
    #                                   # ('clf', LogisticRegression())
    #                                   ])
    
    # TODO: Numerical implementation
    # fraud = FraudData(transaction_file='train_transaction.csv',
    #                   identity_file='train_identity.csv')
    # fraud.data.fillna('-999', inplace=True)
    #
    # object_types = fraud.data.dtypes[fraud.data.dtypes == 'object'].index
    #
    # fraud.data.drop(object_types, axis=1, inplace=True)
    # target = fraud.data['isFraud']
    # fraud.data.drop('isFraud', axis=1, inplace=True)
    #
    # X_train, X_test, y_train, y_test = train_test_split(fraud.data, target,
    #                                                     stratify=target, random_state=42)

    # TODO: This is the model for only Categorical data
    data = DataSplitter(transaction_file='train_transaction.csv', identity_file='train_identity.csv')
    fraud_transformer = FraudFeatureExtractor()
    X_train = fraud_transformer.fit_transform(data.X_train)

    # apply classifier
    model = FraudDetector()
    model.fit(X_train, data.y_train)
    X_test = fraud_transformer.transform(data.X_test)

    y_predictions = model.predict(X_test)[:, 1]
    print('AUC: ', roc_auc_score(data.y_test, y_predictions))
    del X_train
    del X_test
    del data
    # TODO: previous implementation
    # model.submit(X_test)
    # fraud = FraudData(transaction_file='train_transaction.csv',
    #                   identity_file='train_identity.csv')
    # fraud.data.fillna('Unknown', inplace=True)
    #
    # object_types = fraud.data.dtypes[fraud.data.dtypes == 'object'].index
    #
    # # fraud.data.drop(object_types, axis=1, inplace=True)
    # target = fraud.data['isFraud']
    # fraud.data.drop('isFraud', axis=1, inplace=True)
    #
    # X_train, X_test, y_train, y_test = train_test_split(fraud.data, target,
    #                                                     stratify=target, random_state=42)
    # fraud_transformer = FraudFeatureExtractor()
