import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from os import path


data_folder = path.join(path.dirname(path.realpath(__file__)), '../data/input_data/ieee-fraud-detection/')

transaction_features = ['TransactionID', 'ProductCD',
                        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                        'isFraud']

identity_features = ['TransactionID', 'DeviceType', 'DeviceInfo'] + [f'id_{k}' for k in range(12, 39)]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract relevant features for training

    """
    def __init__(self, features=[]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.features]


class TransactionData(object):
    """
    Data Class for Transaction Fraud Data

    """
    def __init__(self, file_name=None):
        self.data = pandas.read_csv(data_folder + file_name)
        self.data = FeatureExtractor(transaction_features).transform(self.data)


class IdentityData(object):
    """
    Data Class for Identity Fraud Data

    """
    def __init__(self, file_name=None):
        self.data = pandas.read_csv(data_folder + file_name)
        self.data = FeatureExtractor(identity_features).transform(self.data)


class FraudData(object):
    """
    Fraud Data class by combining Transaction and Identity data

    """
    def __init__(self, transaction_file=None, identity_file=None):
        self.trans_data = TransactionData(transaction_file).data
        self.identity_data = IdentityData(identity_file).data


if __name__ == '__main__':
    # trans_data = TransactionData('train_transaction.csv')
    identity_data = IdentityData('train_identity.csv')
