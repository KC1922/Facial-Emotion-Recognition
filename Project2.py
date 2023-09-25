import csv
import numpy as np
import argparse

from evalMetrics import printEvalMetrics
from crossValidation import crossFoldValidation
from featureExtraction import computeFeatures
from plotFeatures import plotFeatureBoxplots, plotAllBoxplots

#parse in command line arguments
parser = argparse.ArgumentParser(description='Project2 for physiological data classification')
parser.add_argument('dataType', type=str, help='Data type (dia, sys, eda, res, or all)')
parser.add_argument('dataFile', type=str, help='Absolute directory of the data file')
args = parser.parse_args()

dataType = args.dataType
experiment = args.dataType + '.txt'

#define a dictionary to map data type to column name in the data file 
dataTypeMap = {
    'dia': 'BP Dia_mmHg',
    'sys': 'LA Systolic BP_mmHg',
    'eda': 'EDA_microsiemens',
    'res': 'Respiration Rate_BPM',
    'all': None
}

#read the CSV file using the csv module
rows = []
max_columns = 0

print("Reading data from " + args.dataFile + "...\n")

with open(args.dataFile, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        max_columns = max(max_columns, len(row))
        rows.append(row)

#pad the rows with None values so that all rows have the same number of columns (ensures numpy array can be created)
for row in rows:
    row += [None] * (max_columns - len(row))

#convert rows to a numpy array
data = np.array(rows, dtype=object)

#filter the numpy array based on the data type based on the dataTypeMap dictionary
if dataTypeMap[dataType] is not None:
    dataFiltered = data[data[:, 1] == dataTypeMap[dataType]]
else:
    dataFiltered = data

print("File read successfully! Creating features...\n")

#create a features array based on the data type, will either be 4 or 16 elements
if dataType == 'all':
    featuresFusion = []
    for dt in ['dia', 'sys', 'eda', 'res']:
        dataFiltered = data[data[:, 1] == dataTypeMap[dt]]
        featuresDataType = np.apply_along_axis(computeFeatures, 1, dataFiltered[:, 3:].astype(float))
        featuresFusion.append(featuresDataType)
    features = np.concatenate(featuresFusion, axis=1)
else:
    features = np.apply_along_axis(computeFeatures, 1, dataFiltered[:, 3:].astype(float))


#add the labels (pain or no pain) and subject labels to the features array
subjectLabels = dataFiltered[:, :2]
labels = np.where(dataFiltered[:, 2] == "No Pain", 0, 1).astype(int)
featuresData = np.hstack((subjectLabels, labels.reshape(-1, 1), features))

print("Features created successfully! Calling crossFoldValidation...\n")

#call the crossFoldValidation function with the chosen classifier and featuresData
predictions, testIndices, y = crossFoldValidation(featuresData)
testIndices = np.array(testIndices, dtype=int)

#call the printEvalMetrics function to print the confusion matrix, accuracy, precision, and recall
print("Cross fold validation complete! Printing results...\n")
printEvalMetrics(predictions, testIndices, y, experiment)

#plot feature boxplots
if(dataType != 'all'):
    plotFeatureBoxplots(featuresData, experiment)
else:
    plotAllBoxplots(featuresData, experiment)