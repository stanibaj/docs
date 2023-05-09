Examples
=========

Example 1
----------
This example uses the Breast Cancer Wisconsin (Diagnostic) Data Set that consists of 569 data points with 32 float atributes and binary label to be predicted. The prediction is aimed to define whether the beast cancer tumor is malignant or benign.

.. code-block::
    :caption: Example usage of GaussianNaiveBayesWithSlidingWindow model with evaluation of predicted result based on metrics *Accuracy, Precission, Recall and F1-score*.

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
    import numpy as np

    # load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # split the data into test and train sets using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    # create the GaussianNaiveBayesWithSlidingWindow model
    nb = GaussianNaiveBayesWithSlidingWindow()

    # train the model
    for xi, yi in zip(X_train, y_train):
        nb.learn_one(xi, yi)


    # let the model predict the labels of randomly generated float data points
    pred_arr = []
    truth_y = []
    for xi, yi in zip(X_test, y_test):
        pred = nb.predict_one(xi)
        pred_arr.append(pred)
        truth_y.append(yi)
        
    # compute the metrics and print them
    accuracy = accuracy_score(truth_y, pred_arr)
    precision = precision_score(truth_y, pred_arr)
    recall = recall_score(truth_y, pred_arr)
    f1 = f1_score(truth_y, pred_arr)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")


Example 2
----------

This example tests the implementation of *var_smoothing* parameter, which should add the Gaussian Naive
Bayes classifier numerical stability when features with zero variance are present. The example show it by
generating 100 data points where each data point is defined by 3 float features. Feature 0 is the feature with
zero variance. Other two features are random float. There is a rule that defines the label of each data point.
The rule says that if the data point has value of feature 1 higher than 3 and the value of feature 2 lower
than 2, then this data point is labelled as 1, otherwise it is labelled as 0. After generate the data points
there is the learing phase, predicting phase and calculating metrics phase.

.. code-block::
    :caption: Example of GaussianNaiveBayesWithSlidingWindow model predicting labels on data with one feature with zero variance.

    # Generate a dataset with a 3 features.
    X = np.random.rand(100, 3)

    # Set the value of feature 0 to be constant to simulate zero variance.
    X[:, 0] = np.mean(X[:, 0])
    
    # Set the first half of the data points to be label 1.
    X[:50, 1] = np.random.uniform(3.1,10.0,size=(50,))
    X[:50, 2] = np.random.uniform(-5.0,1.9,size=(50,))

    # Set the second half of the data points to be random from range [-10.0,10> .
    X[50:, 1] = np.random.uniform(-10.0,10.0,size=(50,))
    X[50:, 2] = np.random.uniform(-10.0,10.0,size=(50,))

    # Shuffle the data points.
    np.random.shuffle(X)

    # Label the data points:
    #   if feature 1 is greater than 3 and feature 2 is lower than 2 => label is 1
    #   else label is 0
    y = []
    for i in range(len(X)):
        y_val = 1 if X[i][1] > 3 and X[i][2] < 2 else 0
        y.append(y_val)

    # Split the data into test and train sets using train_test_split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    print(y_train.count(1))
    print(y_train.count(0))

    # Initialize the classifier with window size 10 and default value of var_smoothing parameter.
    clf = GaussianNaiveBayesWithSlidingWindow(window_size=10)

    # Train the classifier with the dataset.
    for i in range(len(X_train)):
        clf.learn_one(X_train[i], y_train[i])

    # Let the model predict the labels.
    pred_arr = []
    truth_y = []
    for xi, yi in zip(X_test, y_test):
        pred = clf.predict_one(xi)
        pred_arr.append(pred)
        truth_y.append(yi)

    # Compute the metrics and print them.
    accuracy = accuracy_score(truth_y, pred_arr)
    precision = precision_score(truth_y, pred_arr)
    recall = recall_score(truth_y, pred_arr)
    f1 = f1_score(truth_y, pred_arr)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")


Example 3
----------

The purpose of this example is to show the sliding window that enhances the Gaussian Naive Bayes classifier
with the mechanism that allows the model to adapt to changes in data distribution over time. In the example we have
100 datapoints of 3 features in the training set. Then there is anothr 100 points data stream generated but
the feature 1 of each generated datapoint is gradually increased by a small ammount. This leads to the mean of the
feature 1 to be gradually increased over time. The model learns from the data stream one data point at a time. 
At the same time the model predict the value of another data point generated the same way.  

.. code-block::
    :caption: Example of GaussianNaiveBayesWithSlidingWindow model predicting labels of data that are gradually changing.

    import numpy as np
    from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    # Define initial parameters
    window_size = 20
    var_smoothing = 1e-9

    # Create a Gaussian Naive Bayes classifier with sliding window
    gnb = GaussianNaiveBayesWithSlidingWindow(window_size=window_size, var_smoothing=var_smoothing)

    # Define number of samples and features
    n_samples = 100
    n_features = 3

    # Generate initial dataset
    X = np.random.rand(n_samples, n_features) * 20 - 10
    y = []
    # Generate initial labels
    for x in X:
        y.append(int(x[1] > 0))

    # Train the classifier with the initial dataset
    for i in range(n_samples):
        gnb.learn_one(X[i], y[i])

    # Calculate current mean of feature 1
    sum_feature_1 = np.sum(X[:][1])
    mean_feature_1 = sum_feature_1 /n_samples
    print("Mean of feature 1 : " + str(mean_feature_1))

    pred_arr = []
    truth_y = []
    # Gradually change the mean of feature 1 over time
    for i in range(n_samples, n_samples * 2):
        # Generate new data point from range (-10,10)
        X_new = np.random.rand(n_features) * 20 - 10
        # Gradually changing mean of feature 1
        X_new += np.array([0, i/n_samples * 6.9, 0])
        # Assign label based on a threshold of feature 1
        y_new = int(X_new[1] > 5)
        # Train the classifier with the new data point
        gnb.learn_one(X_new, y_new)
        # Generate test data point
        X_test = np.random.rand(n_features)  * 20 - 10 + np.array([0, i/n_samples * 6.9, 0])
        y_test = int(X_test[1] > 5)

        # Calculate current mean of feature 1
        sum_feature_1 += X_new[1]
        mean_feature_1 = sum_feature_1 / i
        print("\n Mean of feature 1 : " + str(mean_feature_1))

        # Predict the label of a test data point
        y_pred = gnb.predict_one(X_test)
        pred_arr.append(y_pred)
        truth_y.append(y_test)
    # Compute the metrics and print them.
    accuracy = accuracy_score(truth_y, pred_arr)
    precision = precision_score(truth_y, pred_arr)
    recall = recall_score(truth_y, pred_arr)
    f1 = f1_score(truth_y, pred_arr)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}") 
