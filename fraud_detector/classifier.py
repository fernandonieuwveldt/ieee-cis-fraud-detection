from detector import FraudData
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from os import path
from constants import emails
import pandas
from fraud_detector import INPUT_DIR
from fraud_detector import OUTPUT_DIR


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

    def transform(self, X, y=None):
        important_features = ['card4', 'card6', 'P_emaildomain', 'ProductCD']
        X = X[important_features]
        # X.loc[:, 'P_emaildomain'] = X.loc[:, 'P_emaildomain'].map(emails)
        # X = X['P_emaildomain']
        return X.fillna('Unknown')


class FraudDetector(BaseEstimator):
    """
    Fraud detector model

    """
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.classifier = None  # this will be another sklearn estimator

    def fit(self, xtrain=None, ytrain=None):
        """
        Fit model on FraudData

        :param xtrain: table with features
        :param ytrain: target variable
        :return:
        """
        self.classifier = Pipeline(steps=[('scaler', StandardScaler()),
                                          ('clf', RandomForestClassifier(n_estimators=self.n_estimators))
                                          ]
                                   )
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

    fraud = FraudData(transaction_file='train_transaction.csv',
                      identity_file='train_identity.csv')
    fraud.data.fillna(-999, inplace=True)

    object_types = fraud.data.dtypes[fraud.data.dtypes == 'object'].index

    fraud.data.drop(object_types, axis=1, inplace=True)
    target = fraud.data['isFraud']
    fraud.data.drop('isFraud', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(fraud.data, target,
                                                        stratify=target, random_state=42)
    # apply classifier
    model = FraudDetector()
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)[:, 1]
    print('AUC: ', roc_auc_score(y_test, y_predictions))
    model.submit(X_test)
