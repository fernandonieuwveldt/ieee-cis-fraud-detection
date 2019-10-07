from constants import INPUT_DIR, OUTPUT_DIR, emails, dev_info_map, transaction_features, counting_variables_numeric, \
                      counting_variables_categoric, m_features, time_delta_variables, identity_features, \
                      set_counting_types, vesta
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from transformers import NumericalTransformer, CategoricalTransformer, GroupSelector, DropMissingFeatures, \
    EmbeddingTransformer, MultiColumnLabelEncoder


class FraudFeatureExtractor(TransformerMixin):
    """
    Data transforming pipeline

    """
    def __init__(self, with_embedding=False):
        self.with_embedding = with_embedding
        self.mapper = None
        self.dropper = None

    def fraud_detector_extractor_mapper(self):
        """
        Set up pipeline to transform data

        :return: sklearn.pipeline object
        """
        pipeline_steps = []

        categorical_pipeline = Pipeline(steps=[('categorical_features', CategoricalTransformer()),
                                               ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))
                                               ])
        pipeline_steps.append(('categorical_pipeline', categorical_pipeline))

        if self.with_embedding:
            embedding_pipeline = Pipeline(steps=[('categorical_features', EmbeddingTransformer()),
                                                 ('encode_features', MultiColumnLabelEncoder())])
            pipeline_steps.append(('embedding_pipeline', embedding_pipeline))

        numerical_pipeline = Pipeline(steps=[('numeric_features', NumericalTransformer()),
                                             ('imputer', SimpleImputer(strategy='mean')),
                                             ('scaler', StandardScaler()),
                                             ])
        pipeline_steps.append(('numerical_pipeline', numerical_pipeline))

        vesta_pipeline = Pipeline(steps=[('vesta_features', GroupSelector(features=vesta)),
                                         ('imputer', SimpleImputer(strategy='mean')),
                                         ('pca', PCA(n_components=10, whiten=True)),
                                         ])
        pipeline_steps.append(('vesta_pipeline', vesta_pipeline))

        timedelta_pipeline = Pipeline(steps=[('timedelta_features', GroupSelector(features=time_delta_variables)),
                                             ('imputer', SimpleImputer(strategy='mean')),
                                             ('pca', PCA(n_components=3, whiten=True)),
                                             ])
        pipeline_steps.append(('timedelta_pipeline', timedelta_pipeline))

        transformer_pipeline = FeatureUnion(transformer_list=pipeline_steps)
        return transformer_pipeline

    def fit(self, samples, target):
        """
        Apply transformer to drop features with too many missing values and than fit the pipeline

        :param samples: dataframe with observations X features
        :param target: numpy array with targets
        :return: FraudFeatureExtractor object
        """
        self.dropper = DropMissingFeatures()
        self.dropper.fit(samples)
        features_dropped = self.dropper.transform(samples)
        self.mapper = self.fraud_detector_extractor_mapper().fit(features_dropped, target)
        return self

    def transform(self, x=None):
        """
        Transform data based on the transformer pipeline

        :param x:  pandas dataframe with the training or validation data
        :return: transformed data_frame
        """
        return self.mapper.transform(self.dropper.transform(x))
