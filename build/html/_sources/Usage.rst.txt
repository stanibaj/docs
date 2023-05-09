Usage
=====

Installation
------------

To use GaussianNaiveBayesWithSlidingWindow, first clone it into your directory using *curl* command:

.. code-block:: console

   $ curl https://raw.githubusercontent.com/stanibaj/GNBwSWClassifier/main/GNBwSWClassifier.py -o GNBwSWClassifier.py

After the *curl* is done, just include the cloned python module into your python file using *import* directive:

.. code-block::

    from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow

Example
------------

.. code-block::
    :caption: Example usage of GaussianNaiveBayesWithSlidingWindow model on 1000 datapoints of 3 float features with binary label.

    from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
    import numpy as np
    from sklearn.model_selection import train_test_split

    #define the number of samples in training set
    train_size = 1000
    test_size = int(train_size * 0.2)

    # generate some sample data of three float features
    X = np.random.rand(train_size, 3)
    y = np.random.randint(0, 2, train_size)

    # split the data into test and train sets using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    # create the GaussianNaiveBayesWithSlidingWindow model
    nb = GaussianNaiveBayesWithSlidingWindow()

    # train the model
    for xi, yi in zip(X_train, y_train):
        nb.learn_one(xi, yi)

    # let the model predict the labels of randomly generated float data points
    pred_arr = []
    for xi, yi in zip(X_test, y_test):
        pred = nb.predict_one(xi)
        pred_arr.append(pred)

    # print the prediction array into console
    print(pred_arr)