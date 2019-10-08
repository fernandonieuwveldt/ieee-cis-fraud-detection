from constants import INPUT_DIR, OUTPUT_DIR, emails, dev_info_map, transaction_features, counting_variables_numeric, \
                      counting_variables_categoric, m_features, time_delta_variables, identity_features, \
                      set_counting_types, set_v_types
import pandas
import pandas_profiling
from sklearn.base import TransformerMixin
from copy import deepcopy
from sklearn.model_selection import train_test_split
from pre_process import PreProcessIdentity, PreProcessTransactions


class FeatureExtractor(TransformerMixin):
    """
    Extract relevant features for training

    """
    def __init__(self, features=None):
        self.features = deepcopy(features)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'isFraud' not in X.columns and self.features == transaction_features:  # hack for test data
            self.features.remove('isFraud')
        return X[self.features]


class TransactionData(object):
    """
    Data Class for Transaction Fraud Data

    """

    def __init__(self, file_name=None):
        self.pre_process = PreProcessTransactions()
        # transaction data file is BIG so read in by chunk
        self.data = pandas.concat([FeatureExtractor(transaction_features).transform(chunk)
                                   for chunk in pandas.read_csv(INPUT_DIR + file_name, chunksize=10000)])
        self.data = self.pre_process.transform(self.data)
        self.data = self.data.astype(set_v_types)
        self.data.set_index('TransactionID')


class IdentityData(object):
    """
    Data Class for Identity Fraud Data

    """
    def __init__(self, file_name=None):
        self.pre_process = PreProcessIdentity()
        self.data = pandas.read_csv(INPUT_DIR + file_name)
        self.data = FeatureExtractor(identity_features).transform(self.data)
        self.data = self.pre_process.transform(self.data)

        self.data.set_index('TransactionID')


class FraudData(object):
    """
    Fraud Data class by combining Transaction and Identity data

    """
    def __init__(self, transaction_file=None, identity_file=None):
        transaction_data = TransactionData(transaction_file).data
        identity_data = IdentityData(identity_file).data
        self.data = pandas.merge(transaction_data, identity_data, how='left', on='TransactionID')

        del transaction_data
        del identity_data

    def profile_report(self, df_tran, df_id):
        """
        Create simple EDA report

        :param df_tran: pandas data_frame for transaction data
        :param df_id: pandas data_frame for identity data
        :return: saved html file for the report
        """
        self._gen_profile(df_tran, output_file='transaction_profile.html')
        self._gen_profile(df_id, output_file='identity_profile.html')
        self._gen_profile(self.data, output_file='full_profile.html')

    @staticmethod
    def _gen_profile(data_frame=None, output_file=None):
        pandas_profiling.ProfileReport(data_frame).to_file(OUTPUT_DIR + output_file)


class DataSplitter(object):
    """
    Simple data object to store all training data information

    """
    def __init__(self, transaction_file='', identity_file=''):
        fraud = FraudData(transaction_file=transaction_file,
                          identity_file=identity_file)
        features = fraud.data.drop('isFraud', axis=1)
        target = fraud.data['isFraud'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target,
                                                                                stratify=target, random_state=42)

        # free up memory
        del fraud
        del features
        del target


if __name__ == '__main__':
    fraud = FraudData(transaction_file='train_transaction_5000.csv',
                      identity_file='train_identity_5000.csv')
