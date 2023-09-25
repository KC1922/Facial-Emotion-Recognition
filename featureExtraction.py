import numpy as np

def computeFeatures(arr):
    #perform feature extraction on the data by calculating the mean, variance, min, and max
    #for any None values, replace with np.nan so they are ignored in the calculations
    arr = np.array([float(x) if x != 'None' else np.nan for x in arr])
    return np.array([np.nanmean(arr), np.nanvar(arr), np.nanmin(arr), np.nanmax(arr)])


