from constants import INPUT_DIR, OUTPUT_DIR, emails, dev_info_map, transaction_features, counting_variables_numeric, \
                      counting_variables_categoric, m_features, time_delta_variables, identity_features, \
                      set_counting_types, vesta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from pre_process import reduce_mem_usage
import numpy
from sklearn.decomposition import PCA


class DropMissingFeatures(TransformerMixin):
    """
    Transformer drops features with too many missing values, i.e. gt drop_threshold

    """

    def __init__(self, drop_threshold=1.0):
        """
        constructor

        """
        self.drop_threshold = drop_threshold
        self.keep_features = None

    def fit(self, X, y=None):
        """
        Fits and sets the keep features attribute

        :param X: dataframe with observations by features
        :param y: None
        :return: features to be dropped
        """
        self.keep_features = list(X.columns[X.isnull().mean() <= self.drop_threshold])
        return self

    def transform(self, X, y=None):
        """
        Extracts only relevant features

        :param X: data frame with observations X features
        :param y: None
        :return: data frame
        """
        return X[self.keep_features]


class TypeSelector(TransformerMixin):
    """
    Transformer that select columns based on type or from list of names

    """
    def __init__(self, type=None):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Select only features of specified type

        :param X: pandas data_frame
        :param y: None
        :return: pandas data_frame only containing features of specified type
        """
        features = X.dtypes[X.dtypes == self.type].index
        return X[features]


class CountingTransform(TransformerMixin):
    def __init__(self):
        """
        constructor
        """

    def fit(self):
        return self

    def transform(self, xcount=None):
        """
        Some counting features are skewed and will be regarded as categorical

        :param xcount: data frame with observations X features(counting features)
        :return: data frame with engineered groups
        """
        for cvar in counting_variables_categoric:
            xcount.loc[xcount.loc[:, cvar] >= 2, cvar] = 2
        return xcount


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer extracting categorical features and engineer new ones

    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, xc=None):
        xc_copy = xc.copy()
        x_snipped = TypeSelector(type='object').transform(xc_copy).fillna('Unknown').astype('str')
        x_snipped = reduce_mem_usage(x_snipped)
        return x_snipped


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer extracting embedding candidate features

    """
    N_UNIQUE = 4

    def __init__(self):
        self.candidates_features = None

    def fit(self, X, y=None):
        """
        Get list of features that are embedding feature candidates

        :param X: pandas data_frame
        :param y: None
        :return: EmbeddingTransformer object
        """
        x_snipped = TypeSelector(type='object').transform(X).fillna('Unknown')

        candidates = x_snipped.nunique() > self.N_UNIQUE

        self.candidates_features = list(candidates[candidates].index)
        return self

    def transform(self, xc=None):
        return xc[self.candidates_features].fillna('Unknown')


class MultiColumnLabelEncoder(TransformerMixin):
    """
    Apply LabelEncoder on multiple variables.

    Sets a default label if unseen labels are supplied during inference
    """
    def __init__(self):
        self.encoders = None
        self.embedding_candidates = []

    def fit(self, X, y=None):
        """
        Fit LabelEncoder's on multiple features

        :param X: pandas data_frame
        :param y: None
        :return: MultiColumnLabelEncoder object
        """
        self.encoders = {}
        for feature, c in X.iteritems():
            encoder = LabelEncoder()
            encoder.fit(c)
            self.encoders[feature] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            self.embedding_candidates.append(feature)

        return self

    def transform(self, X):
        """
        Apply LabelEncoder and supply default values for unseen labels during inference

        :param X: pandas data frame
        :return: label encoded pandas data frame
        """
        for feature in self.embedding_candidates:
            X.loc[:, feature] = X.loc[:, feature].apply(lambda x: self.encoders[feature].get(x, 999))
        return X


class NumericalTransformer(TransformerMixin):
    """
    Transformer extracting numerical features and feature engineering

    """
    cards = ['card1', 'card2', 'card3', 'card5']

    def __init__(self):
        self.count_mapper = {}
        self.fraud_mapper = {}

    def fit(self, X, y=None):
        """
        Fit counting mappers

        :param X: pandas data_frame
        :param y: None
        :return: NumericalTransformer object
        """
        for card in self.cards:
            counter = X[card].value_counts()
            fraud_counter = X[card][y == 1].value_counts()
            self.count_mapper[card] = dict(zip(counter.index, counter.values))
            self.fraud_mapper[card] = dict(zip(fraud_counter.index, fraud_counter.values))
        return self

    def transform(self, xn=None):
        xn_copy = xn.copy()
        for card in self.cards:
            xn_copy[card] = xn_copy[card].map(self.count_mapper[card]).fillna(1.0)
            xn_copy[card+'_fraudcases'] = xn_copy[card].map(self.fraud_mapper[card]).fillna(1.0)
            xn_copy[card+'_proportion'] = xn_copy[card+'_fraudcases'] / xn_copy[card]

        object_types = list(xn_copy.dtypes[xn_copy.dtypes == 'object'].index)
        # drop features that will be handled by a different pipeline
        xn_copy.drop(object_types, axis=1, inplace=True)
        xn_copy.drop(vesta, axis=1, inplace=True)
        xn_copy.drop(time_delta_variables, axis=1, inplace=True)
        xn_copy = reduce_mem_usage(xn_copy)
        return xn_copy


class GroupSelector(TransformerMixin):
    """
    Selects only supplied features.

    This is a helper class to be able select features in sklearn.pipeline
    """
    def __init__(self, features=None):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, xv=None):
        return xv[self.features]


class AutoPCA(TransformerMixin):
    """
    Selects the optimal n_components to use based on the explained_variance threshold.

    """
    def __init__(self, variance_threshold=0.98):
        self.opt_ncomponents = None
        self.variance_threshold = variance_threshold
        self.pca = PCA()

    def fit(self, X, y=None):
        """
        Apply PCA and get optimal number of components based on the variance

        :param X: feature matrix
        :param y: None
        :return: self return object
        """
        self.pca.fit(X)
        self.opt_ncomponents = numpy.argmax(self.pca.explained_variance_ratio_.cumsum() > self.variance_threshold)
        return self

    def transform(self, X, y=None):
        return self.pca.transform(X)[:, :self.opt_ncomponents]

