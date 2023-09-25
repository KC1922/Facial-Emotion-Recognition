import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn import svm

def crossFoldValidation (data):
    #code is mostly the same from project 1
    print("Starting cross fold validation...")
    X = np.array(data[:, 3:], dtype=float)
    y = np.array(data[:, 2], dtype=int)
    s = np.array(data[:, 0]) 

    clf = svm.SVC()
    #use GroupKFold to split the data by subject 
    group_kfold = GroupKFold(n_splits=10)

    pred = []
    testIndicesList = []

    for i, (trainIndex, testIndex) in enumerate(group_kfold.split(X, y, s)):
        #train the classifier on the training data
        clf.fit(X[trainIndex], y[trainIndex])
        pred.append(clf.predict(X[testIndex]))
        testIndicesList.append(testIndex.tolist())
        print("Fold " + str(i+1) + " complete.")

    return pred, testIndicesList, y
