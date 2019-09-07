import pandas
import pandas_profiling
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
from fraud_detector import INPUT_DIR
from fraud_detector import OUTPUT_DIR
from sklearn.model_selection import train_test_split
import os


transaction_features = ['TransactionID', 'ProductCD', 'TransactionAmt', 'dist1', 'dist2',
                        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                        'isFraud']

identity_features = ['TransactionID', 'DeviceType', 'DeviceInfo'] + [f'id_{k}' for k in range(12, 39)]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract relevant features for training

    """
    def __init__(self, features=None):
        self.features = deepcopy(features)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # remove features with too many missing values here
        if 'isFraud' not in X.columns and self.features == transaction_features:  # bad hack for test data
            self.features.remove('isFraud')
        return X[self.features]


class TransactionData(object):
    """
    Data Class for Transaction Fraud Data

    """

    def __init__(self, file_name=None):
        # transaction data file is BIG so read in by chunk
        self.data = pandas.concat([FeatureExtractor(transaction_features).transform(chunk)
                                   for chunk in pandas.read_csv(INPUT_DIR + file_name, chunksize=10000)])
        self.data.set_index('TransactionID')


class IdentityData(object):
    """
    Data Class for Identity Fraud Data

    """
    def __init__(self, file_name=None):
        self.data = pandas.read_csv(INPUT_DIR + file_name)
        self.data = FeatureExtractor(identity_features).transform(self.data)
        self.data.set_index('TransactionID')


class FraudData(object):
    """
    Fraud Data class by combining Transaction and Identity data

    """
    def __init__(self, transaction_file=None, identity_file=None):
        transaction_data = TransactionData(transaction_file).data
        identity_data = IdentityData(identity_file).data
        self.data = pandas.merge(transaction_data,  identity_data, how='left', on='TransactionID')
        del transaction_data
        del identity_data

    def profile_report(self):
        self._gen_profile(self.transaction_data, output_file='transaction_profile.html')
        self._gen_profile(self.identity_data, output_file='identity_profile.html')
        self._gen_profile(self.data, output_file='full_profile.html')

    @staticmethod
    def _gen_profile(data_frame=None, output_file=None):
        pandas_profiling.ProfileReport(data_frame).to_file(OUTPUT_DIR + output_file)


class DataSplitter(object):
    """
    Simple data object to store all training data information

    Args:
        transaction_file: csv file name containing data
        identity_file: csv file name containing data

    Attributes:
        data: pandas dataframe containing all training data
        features: features of the training data
        target: Target variable, client that took up product
        X_train: the data on which model will be trained on
        X_test: the data on which the model will be tested on
        y_train: target variable for training
        y_test: target variable for testing

    """
    def __init__(self, transaction_file='', identity_file=''):
        fraud = FraudData(transaction_file=transaction_file,
                          identity_file=identity_file)
        # fraud.data.fillna('Unknown', inplace=True)
        features = fraud.data.drop('isFraud', axis=1)
        target = fraud.data['isFraud'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target,
                                                                                stratify=target, random_state=42)
        # free up memory
        del fraud
        del features
        del target


if __name__ == '__main__':
    # fraud = FraudData(transaction_file='train_transaction.csv',
    #                   identity_file='train_identity.csv')
    # fraud.profile_report()
    fraud_data = DataSplitter(transaction_file='train_transaction.csv', identity_file='train_identity.csv')
