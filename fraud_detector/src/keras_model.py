import pandas
import tensorflow
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras
from transformers import TypeSelector
from feature_extractor import FraudFeatureExtractor
from utils import cast_to_type

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
config = tensorflow.ConfigProto(device_count={"CPU": 8})
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=config))


class EmbeddingBasedClassifier:
    """
    Model based on embedding layers for categorical features

    """

    EPOCHS = 10
    OPTIMIZER = optimizers.SGD(lr=0.03, nesterov=True)
    EMBEDDING_RATIO = 0.25
    MAX_EMBEDDING = 50
    BATCH_SIZE = 256
    VERBOSE = 0

    ES_PARAMS = {'monitor': 'val_loss',
                 'mode': 'min',
                 'verbose': 1,
                 'patience': 20}

    MC_PARAMS = {'filepath': '../data/saved_models/best_model.h5',
                 'monitor': 'val_loss',
                 'mode': 'min',
                 'verbose': 1,
                 'save_best_only': True}

    def __init__(self):
        self.early_stopping = EarlyStopping(**self.ES_PARAMS)
        self.model_checkpoint = ModelCheckpoint(**self.MC_PARAMS)
        self.transformer = FraudFeatureExtractor(with_embedding=True)
        self.inputs = []
        self.layers = []
        self.classifier = None
        self.model = None
        self.feature_category_mapper = None
        self.feature_mode = None
        self.embedding_output_mapper = None

    def embedding_mapper(self, data_frame=None):
        """
        Extract candidate embedding features and create mapper of number unique entries and mapper for output dimension

        :param data_frame: data frame
        :return: tuple (list of embedding features, dictionary with nr unique entries, output dimension)
        """
        cat_data_frame = TypeSelector(type='object').transform(data_frame)

        numeric_features = list(data_frame.dtypes[data_frame.dtypes == 'float64'].index)
        categoric_features = list(cat_data_frame.columns)
        embedding_features = list(cat_data_frame.loc[:, cat_data_frame.nunique() > 2].columns)

        # remove embedding features from categoric features
        categoric_features = set(categoric_features) - set(embedding_features)

        feature_mode = {feature: cat_data_frame[feature].nunique()+1 for feature in embedding_features}
        embedding_output_mapper = {feature: min(int(feature_mode[feature] * self.EMBEDDING_RATIO)+2, self.MAX_EMBEDDING)
                                   for feature in embedding_features}

        feature_category_mapper = {'numeric_features': numeric_features,
                                   'categoric_features': list(categoric_features),
                                   'embedding_features': embedding_features}

        return feature_category_mapper, feature_mode, embedding_output_mapper

    @staticmethod
    def preproc_embedding_layer(data_frame=None, feature_category_mapper=None):
        """
        Creates new unique ordinal mapping to feed to embedding layer and create proper format for Keras model

        :param data_frame: pandas data frame
        :param feature_category_mapper:
        :return: list of preprocessed data frames
        """
        unique_values = {feature: data_frame[feature].unique() for feature in feature_category_mapper['embedding_features']}
        value_mapper = {ek: dict(map(lambda x: (x[1], x[0]), enumerate(unique_values[ek])))
                        for ek in feature_category_mapper['embedding_features']}

        preproc_embedding = [data_frame[c].map(value_mapper[c]).values for c in feature_category_mapper['embedding_features']]
        preproc_categorical = data_frame.loc[:, feature_category_mapper['categoric_features']].values
        preproc_numerical = data_frame.loc[:, feature_category_mapper['numeric_features']].values

        # unpack
        train_data = []
        for pe in preproc_embedding:
            train_data.append(pe)
        train_data.append(preproc_categorical)
        train_data.append(preproc_numerical)

        return train_data

    @staticmethod
    def create_embedding_layer(n_unique=None, output_dim=None, input_length=1):
        """
        Creates embedding layers

        :param n_unique: dimension of unique labels
        :param output_dim: dimension of embedding matrix
        :param input_length: default 1
        :return: input data info, embedding layer info
        """
        _input = Input(shape=(1, ))
        _embedding = Embedding(n_unique, output_dim, input_length=input_length)(_input)
        _embedding = Reshape(target_shape=(output_dim, ))(_embedding)
        return _input, _embedding

    def load_data(self):
        """
        Loads data

        :return:
        """

    def build_network(self, feature_category_mapper=None, feature_mode=None, embedding_output_mapper=None):
        """
        Build up network with all embedding and other(numeric) layers

        :param feature_category_mapper:
        :param feature_mode:
        :param embedding_output_mapper:
        :return: compiled keras model
        """
        # add embedding layers
        if feature_category_mapper['embedding_features'] is not None:
            for feature in feature_category_mapper['embedding_features']:
                embedding_input, embedding_layer = self.create_embedding_layer(feature_mode[feature],
                                                                               embedding_output_mapper[feature])
                self.inputs.append(embedding_input)
                self.layers.append(embedding_layer)

        # add layer for other categoric features that are not embedding features
        if feature_category_mapper['categoric_features'] is not None:
            categorical_input = Input(shape=(len(feature_category_mapper['categoric_features']), ))
            categoric_layer = Dense(50)(categorical_input)
            self.inputs.append(categorical_input)
            self.layers.append(categoric_layer)

        # add layer for other numeric features
        if feature_category_mapper['numeric_features'] is not None:
            numeric_input = Input(shape=(len(feature_category_mapper['numeric_features']), ))
            numeric_layer = Dense(50)(numeric_input)
            self.inputs.append(numeric_input)
            self.layers.append(numeric_layer)

        x = Concatenate()(self.layers)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(10, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(self.inputs, output)
        model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER,  metrics=['acc'])
        return model

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None):
        """
        Fit neural net model

        :param x_train: pandas data_frame
        :param y_train: training targets
        :param x_val: pandas data_frame
        :param y_val: validation targets
        :return:
        """
        self.transformer.fit(x_train, y_train)
        x_train = self.transformer.transform(x_train)
        x_val = self.transformer.transform(x_val)

        # have to convert numpy to pandas
        onehot_pipe = self.transformer.mapper.transformer_list[0][1][1]
        embedding_pipe = self.transformer.mapper.transformer_list[1][1][1]
        self.nr_cat_features = sum([len(elem) for elem in onehot_pipe.categories_]) + \
                               len(embedding_pipe.embedding_candidates)

        x_train, x_val = cast_to_type(x_train, x_val, self.nr_cat_features)

        self.feature_category_mapper, self.feature_mode, self.embedding_output_mapper = self.embedding_mapper(x_train)

        self.classifier = self.build_network(self.feature_category_mapper,
                                             self.feature_mode,
                                             self.embedding_output_mapper)

        x_train_preproc = self.preproc_embedding_layer(x_train, self.feature_category_mapper)
        x_val_preproc = self.preproc_embedding_layer(x_val, self.feature_category_mapper)

        params = {'x': x_train_preproc,
                  'y': y_train,
                  'validation_data': (x_val_preproc, y_val),
                  'batch_size': self.BATCH_SIZE,
                  'epochs': self.EPOCHS,
                  'verbose': self.VERBOSE,
                  'callbacks': [self.early_stopping, self.model_checkpoint]}

        self.classifier.fit(**params)

    def predict(self, x_test=None):
        """
        Loads best model for prediction

        :param x_test:
        :return:
        """
        x_test = self.transformer.transform(x_test)
        x_test, _ = cast_to_type(x_test, x_test, self.nr_cat_features)

        x_test_preproc = self.preproc_embedding_layer(x_test, self.feature_category_mapper)
        saved_model = load_model(self.MC_PARAMS['filepath'])
        return saved_model.predict(x_test_preproc)

    def submit(self, xtest=None, trans_ids=None):
        """
        Saves prediction outputs in kaggle submission file format

        :param xtest: competition test data
        :param trans_ids: transaction ids
        :return: saved kaggle file format
        """
        test_predictions = self.predict(xtest)
        my_submission_file = pandas.DataFrame()
        my_submission_file['TransactionID'] = trans_ids
        my_submission_file['isFraud'] = test_predictions
        my_submission_file.to_csv('../data/output_data/submission.csv', index=False)
