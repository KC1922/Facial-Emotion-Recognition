import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def printEvalMetrics(pred, indicesList, y, filename):
    #code is mostly the same from project 1
    finalPred = []
    groundTruth = []

    for p in pred:
        finalPred.extend(p)
    for i in indicesList:
        groundTruth.extend(y[i])
    #make folder to store experiment results, if it doesn't already exist
    experiment_dir = os.path.join(os.getcwd(), "Experiments")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    with open(os.path.join(experiment_dir, filename), 'w') as f:
        f.write(str(confusion_matrix(finalPred, groundTruth)) + '\n')
        f.write("Precision: " + str(precision_score(groundTruth, finalPred, average='macro')) + '\n')
        f.write("Recall: " + str(recall_score(groundTruth, finalPred, average='macro')) + '\n')
        f.write("Accuracy: " + str(accuracy_score(groundTruth, finalPred)) + '\n')
        f.write("Classifier Score: " + str(f1_score(groundTruth, finalPred)) + '\n')
    print("Experiment results printed to " + os.path.join(experiment_dir, filename) + "\n")
