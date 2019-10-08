Code was used for ieee-fraud-detection competition at kaggle

To run execute below under cd fraud_detector/src directory

## Load the data
```python
    from workflows import *
    # load a snippet of the original data set
    data = DataSplitter(transaction_file='train_transaction_5000.csv', identity_file='train_identity_5000.csv')

```

## Two classes of models exist
* Two classes of models exist, one set for tree based models which can be found in classifier.py: 
    * Tree based models: BaggedRFModel, CatBoostModel, XGBModel, KFoldModel, ClusteredXGBModel
    * The other is Embedding based model using keras. The keras model can be found in keras_model.py
    

# Example of running one of the tree models:
```python
    estimator = BaggedRFModel()
    trained_estimator = train_tree(train_data=data, estimator=estimator)
    trained_estimator.submit(data.X_test)

```

# And running the Embedding based model
```python
    trained_net = train_embedding_model(train_data=data)
    trained_net.submit(data.X_test, data.X_test['TransactionID'])
```
