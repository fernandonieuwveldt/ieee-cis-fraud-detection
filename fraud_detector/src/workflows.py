from classifier import BaggedRFModel, CatBoostModel, XGBModel, KFoldModel, ClusteredXGBModel
from keras_model import EmbeddingBasedClassifier
from fraud import FraudData, DataSplitter
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def train_tree(train_data=None, estimator=CatBoostModel()):
    """
    Train model on data

    :param train_data: FraudData object
    :param estimator: estimator to be fitted(BaggedRFModel by default
    :return: fitted model
    """
    # fit classifier
    estimator.fit(xtrain=train_data.X_train, ytrain=train_data.y_train)

    y_predictions = estimator.predict(train_data.X_test)[:, 1]
    print('AUC on Train set: ', roc_auc_score(train_data.y_train, estimator.predict(train_data.X_train)[:, 1]))
    print('AUC on Test set: ', roc_auc_score(train_data.y_test, y_predictions))
    return estimator


def train_embedding_model(train_data=None):
    """
    Fits embedding based model

    :param train_data: FraudData object
    :return: fitted model
    """


def model_check(scores, y_true):
    """
    Checks how well our model works on test set using a ROC curve
    :param scores: class probabilities from model
    :param y_true: true labels
    :return: ROC curve plot with AUC value
    """
    auc = round(roc_auc_score(y_true, scores), 3)
    fpr, tpr, threshold = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, label="AUC on Test set =" + str(auc))
    plt.plot([[0, 0], [1, 1]])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    data = DataSplitter(transaction_file='train_transaction_5000.csv', identity_file='train_identity_5000.csv')
    estimator = BaggedRFModel()
    trained_estimator = train_tree(data=data, estimator=estimator)
