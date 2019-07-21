import pandas
import pandas_profiling
from sklearn.base import BaseEstimator, TransformerMixin
from os import path
from copy import deepcopy


input_data_folder = path.join(path.dirname(path.realpath(__file__)), '../data/input_data/ieee-fraud-detection/')
output_data_folder = path.join(path.dirname(path.realpath(__file__)), '../data/output_data/')

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
        self.features = deepcopy(features)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # remove features with too many missing values here
        if 'isFraud' not in X.columns and self.features == transaction_features:
            self.features.remove('isFraud')
        return X[self.features]


class TransactionData(object):
    """
    Data Class for Transaction Fraud Data

    """

    def __init__(self, file_name=None):
        # transaction data file is BIG so read in by chunk
        self.data = pandas.concat([FeatureExtractor(transaction_features).transform(chunk)
                                   for chunk in pandas.read_csv(input_data_folder + file_name, chunksize=10000)])
        self.data.set_index('TransactionID')


class IdentityData(object):
    """
    Data Class for Identity Fraud Data

    """
    def __init__(self, file_name=None):
        self.data = pandas.read_csv(input_data_folder + file_name)
        self.data = FeatureExtractor(identity_features).transform(self.data)
        self.data.set_index('TransactionID')


class FraudData(object):
    """
    Fraud Data class by combining Transaction and Identity data

    """
    def __init__(self, transaction_file=None, identity_file=None):
        # self.transaction_data = TransactionData(transaction_file).data
        # self.identity_data = IdentityData(identity_file).data
        # self.data = self.transaction_data.join(self.identity_data, lsuffix='_Transaction', rsuffix='_Identity')
        transaction_data = TransactionData(transaction_file).data
        identity_data = IdentityData(identity_file).data
        self.data = pandas.merge(transaction_data,  identity_data, how='left', on='TransactionID')
        del transaction_data
        del identity_data

    def profile_report(self):
        # self._gen_profile(self.transaction_data, output_file='transaction_profile.html')
        # self._gen_profile(self.identity_data, output_file='identity_profile.html')
        self._gen_profile(self.data, output_file='full_profile.html')

    @staticmethod
    def _gen_profile(data_frame=None, output_file=None):
        pandas_profiling.ProfileReport(data_frame).to_file(output_data_folder+output_file)


if __name__ == '__main__':
    fraud = FraudData(transaction_file='train_transaction.csv',
                      identity_file='train_identity.csv')
    # fraud.profile_report()

