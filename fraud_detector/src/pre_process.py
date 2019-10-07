"""
preprocess data

References from kaggle:

"""
from constants import dev_info_map, emails, counting_variables_categoric, \
    browser_mapper, card_features, risky_addr1, risky_addr2, addr1_mapper, addr2_mapper, vesta, time_delta_variables
from utils import reduce_mem_usage, unique_mapper, make_day_feature, make_hour_feature
from sklearn.base import TransformerMixin
import numpy


class PreProcessIdentity(TransformerMixin):
    """
    Preprocessor for the identity feature matrix

    """
    PLATFORM_MAPPER = {'Android': 1, 'Windows': 2, 'Mac': 3, 'iOS': 3, 'Linux': 4}
    PLATFORM_ID = 'id_30'

    def __init__(self):
        self.features_to_keep = None

    def _os_platform_mapper(self, data_frame=None):
        """
        Apply platform mapping \

        :param data_frame: pandas data_frame
        :return: platformed mapped data_frame
        """
        id_30 = data_frame[self.PLATFORM_ID]
        data_frame.loc[:, self.PLATFORM_ID] = 0

        for key, val in self.PLATFORM_MAPPER.items():
            data_frame.loc[id_30.str.contains(key).fillna(False), self.PLATFORM_ID] = val
        data_frame[self.PLATFORM_ID] = data_frame[self.PLATFORM_ID].astype('object')
        return data_frame

    def set_types(self):
        pass

    @staticmethod
    def remove_features(x=None, nan_threshold=0.4):
        """
        Removes the features with too many missing values
        :param x: data_frame with rows x features
        :param nan_threshold: threshold percentage to remove features with too many missing values
        :return: list of features to keep
        """
        features_to_keep = x.columns[x.isnull().mean() < nan_threshold]
        return list(features_to_keep)

    def fit(self, X=None, y=None):
        """
        return PreProcessIdentity object, no fit for pre-processing

        :param X:
        :param y:
        :return: PreProcessIdentity object
        """
        return self

    def transform(self, xi):
        """
        Apply all preprocessing of the Identity data

        :param xi: pandas data_frame
        :return: processed data_frame
        """
        identity_transformed = xi.copy()
        identity_transformed.loc[:, 'DeviceInfo'] = identity_transformed.loc[:, 'DeviceInfo'].map(dev_info_map)
        identity_transformed = self._os_platform_mapper(identity_transformed)
        identity_transformed['id_31_mapped'] = 0
        identity_transformed['id_31_mapped'] = identity_transformed['id_31'].map(browser_mapper)
        identity_transformed['id_33'] = identity_transformed['id_33'].fillna('-999x-1').map(lambda x: x.split('x')[0]).astype(int) /\
                                        identity_transformed['id_33'].fillna('-999x-1').map(lambda x: x.split('x')[1].strip()).astype(int)
        identity_transformed['id_33'] = identity_transformed['id_33'].replace(999.0, numpy.nan)
        # identity_transformed.drop(['id_30', 'id_31', 'id_33'], axis=1, inplace=True)
        # identity_transformed.drop(['id_31'], axis=1, inplace=True)
        return identity_transformed


class PreProcessTransactions:
    """
    Preprocessor for the transactions feature matrix

    """
    def __init__(self):
        pass

    @staticmethod
    def _counting_transform(xcount=None):
        for cvar in counting_variables_categoric:
            xcount.loc[xcount.loc[:, cvar] >= 2, cvar] = 2
            xcount[cvar] = xcount[cvar].astype('object')
        return xcount

    @staticmethod
    def amount_transforms(xt):
        """
        (See https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering)
        Engineering more features

        :param xt: data frame
        :return: data frame
        """
        xt['TransactionAmt_to_mean_card1'] = xt['TransactionAmt'] / xt.groupby(['card1'])['TransactionAmt'].transform('mean')
        xt['TransactionAmt_to_mean_card4'] = xt['TransactionAmt'] / xt.groupby(['card4'])['TransactionAmt'].transform('mean')
        xt['TransactionAmt_to_std_card1'] = xt['TransactionAmt'] / xt.groupby(['card1'])['TransactionAmt'].transform('std')
        xt['TransactionAmt_to_std_card4'] = xt['TransactionAmt'] / xt.groupby(['card4'])['TransactionAmt'].transform('std')
        xt['TransactionAmt_decimal'] = (xt['TransactionAmt'] - xt['TransactionAmt'].astype(int)).astype(int)

        xt['D15_to_mean_card1'] = xt['D15'] / xt.groupby(['card1'])['D15'].transform('mean')
        xt['D15_to_mean_card4'] = xt['D15'] / xt.groupby(['card4'])['D15'].transform('mean')
        xt['D15_to_std_card1'] = xt['D15'] / xt.groupby(['card1'])['D15'].transform('std')
        xt['D15_to_std_card4'] = xt['D15'] / xt.groupby(['card4'])['D15'].transform('std')

        xt['D15_to_mean_addr1'] = xt['D15'] / xt.groupby(['addr1'])['D15'].transform('mean')
        xt['D15_to_mean_card4'] = xt['D15'] / xt.groupby(['card4'])['D15'].transform('mean')
        xt['D15_to_std_addr1'] = xt['D15'] / xt.groupby(['addr1'])['D15'].transform('std')
        xt['D15_to_std_card4'] = xt['D15'] / xt.groupby(['card4'])['D15'].transform('std')

        return xt.replace(numpy.inf, numpy.nan)

    @staticmethod
    def card_transforms(xt):
        """
        Generate some extra features features based on the card features

        :param xt: pandas data_frame
        :return: pandas data_frame with extra engineered features
        """
        xt['card3'] = xt['card3'].fillna(999).map(int).astype('str')
        xt['card5'] = xt['card5'].fillna(999).map(int).astype('str')
        xt.loc[:, 'addr2_high_risk_87'] = 0
        xt.loc[xt['addr2'] == 87, 'addr2_high_risk_87'] = 1
        xt.loc[:, 'addr2_high_risk_60'] = '0'
        xt.loc[xt['addr2'] == 60, 'addr2_high_risk_60'] = 1
        xt.loc[:, 'addr2_high_risk_96'] = '0'
        xt.loc[xt['addr2'] == 96, 'addr2_high_risk_96'] = 1
        xt.loc[:, 'addr2_high_risk_65'] = '0'
        xt.loc[xt['addr2'] == 65, 'addr2_high_risk_65'] = 1

        return xt

    def transform(self, xt):
        """
        Apply all preprocessing of the data

        :param xt: pandas data_frame
        :return: processed data_frame
        """
        xt = self.amount_transforms(xt)
        xt = self._counting_transform(xt)

        xt['P_isproton'] = (xt['P_emaildomain'] == 'protonmail.com').astype('object')
        xt['R_isproton'] = (xt['R_emaildomain'] == 'protonmail.com').astype('object')

        xt.loc[:, 'P_emaildomain'] = xt.loc[:, 'P_emaildomain'].map(emails)
        xt.loc[:, 'R_emaildomain'] = xt.loc[:, 'R_emaildomain'].map(emails)

        xt['addr1'] = xt['addr1'].fillna(999).map(int).map(addr1_mapper).fillna(999).astype('object')
        xt['addr2'] = xt['addr2'].fillna(999).map(int).map(addr2_mapper).fillna(999).astype('object')

        xt['card1'] = xt['card1'].fillna(999).map(int).astype('float')
        xt['card2'] = xt['card2'].fillna(999).map(int).astype('float')
        xt['card3'] = xt['card3'].fillna(999).map(int).astype('float')
        xt['card5'] = xt['card5'].fillna(999).map(int).astype('float')

        card6_mapper = {'credit': 'credit', 'debit': 'debit', 'debit or credit': 'debit', 'charge card': 'debit'}
        xt['card6'] = xt['card6'].map(card6_mapper)

        xt['day'] = make_day_feature(xt).astype('object')
        xt['hours'] = make_hour_feature(xt).astype('object')

        # drop features no longer needed
        xt.drop(['TransactionDT'], axis=1, inplace=True)

        return xt


