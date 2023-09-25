import numpy as np
import matplotlib.pyplot as plt

def plotFeatureBoxplots(features, fileName):
    meanData = features[:, 3]
    varData = features[:, 4]
    minData = features[:, 5]
    maxData = features[:, 6]

    painMeanData = meanData[features[:, 2] == 1]
    noPainMeanData = meanData[features[:, 2] == 0]

    painVarData = varData[features[:, 2] == 1]
    noPainVarData = varData[features[:, 2] == 0]

    painMinData = minData[features[:, 2] == 1]
    noPainMinData = minData[features[:, 2] == 0]

    painMaxData = maxData[features[:, 2] == 1]
    noPainMaxData = maxData[features[:, 2] == 0]

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)

    axs[0].boxplot([noPainMeanData, painMeanData], vert=False)
    axs[0].set_yticklabels(['No Pain', 'Pain'])
    axs[0].set_title('Mean')

    axs[1].boxplot([noPainVarData, painVarData], vert=False)
    axs[1].set_yticklabels(['No Pain', 'Pain'])
    axs[1].set_title('Variance')

    axs[2].boxplot([noPainMinData, painMinData], vert=False)
    axs[2].set_yticklabels(['No Pain', 'Pain'])
    axs[2].set_title('Min')

    axs[3].boxplot([noPainMaxData, painMaxData], vert=False)
    axs[3].set_yticklabels(['No Pain', 'Pain'])
    axs[3].set_title('Max')

    plt.savefig('Experiments/' + fileName + '.png')

def plotAllBoxplots(features, fileName):
    MEAN_IDX = 3
    VAR_IDX = 4
    MIN_IDX = 5
    MAX_IDX = 6

    NUM_DATATYPES = 4

    meanData = [[] for i in range(NUM_DATATYPES)]
    varData = [[] for i in range(NUM_DATATYPES)]
    minData = [[] for i in range(NUM_DATATYPES)]
    maxData = [[] for i in range(NUM_DATATYPES)]

    for i in range(NUM_DATATYPES):
        meanData[i] = features[:, MEAN_IDX + i * 4]
        varData[i] = features[:, VAR_IDX + i * 4]
        minData[i] = features[:, MIN_IDX + i * 4]
        maxData[i] = features[:, MAX_IDX + i * 4]

        painMeanData = meanData[i][features[:, 2] == 1]
        noPainMeanData = meanData[i][features[:, 2] == 0]

        painVarData = varData[i][features[:, 2] == 1]
        noPainVarData = varData[i][features[:, 2] == 0]

        painMinData = minData[i][features[:, 2] == 1]
        noPainMinData = minData[i][features[:, 2] == 0]

        painMaxData = maxData[i][features[:, 2] == 1]
        noPainMaxData = maxData[i][features[:, 2] == 0]

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5)

        axs[0].boxplot([noPainMeanData, painMeanData], vert=False)
        axs[0].set_yticklabels(['No Pain', 'Pain'])
        axs[0].set_title('Mean')

        axs[1].boxplot([noPainVarData, painVarData], vert=False)
        axs[1].set_yticklabels(['No Pain', 'Pain'])
        axs[1].set_title('Variance')

        axs[2].boxplot([noPainMinData, painMinData], vert=False)
        axs[2].set_yticklabels(['No Pain', 'Pain'])
        axs[2].set_title('Min')

        axs[3].boxplot([noPainMaxData, painMaxData], vert=False)
        axs[3].set_yticklabels(['No Pain', 'Pain'])
        axs[3].set_title('Max')

        plt.savefig(f'Experiments/{fileName}_dataType{i+1}.png')
        plt.close()