import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=(plotSize, plotSize), diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


def plotType1(df):
    df1 = pd.get_dummies(df.Type1)
    s1 = df1.sum(axis=0)
    print(type(s1))
    s1 = s1.sort_values(ascending=False)
    s1.plot.bar(x='lab', y='val', rot=45)
    plt.show()

def plotType1vsType2(df):
    df1 = pd.get_dummies(df.Type1, prefix="Type1_", dummy_na=True)
    df2 = pd.get_dummies(df.Type2, prefix="Type2_", dummy_na=True)
    print(df1.head(2))
    print(df2.head(2))

    df2_list = df2.index.tolist()
    df1_list = df1.index.tolist()
    arr = np.zeros((19,19))
    print(len(df1))
    for i in range(len(df1)):
        # index i = where one-hot type1 is 1
        # index j = where one-hot type2 is 1
        # arr[i,j] += 1
        for idx in range(len(df1.iloc[i])):
            if df1.iloc[i][idx] == 1:
                _i = idx

        for idx in range(len(df2.iloc[i])):
            if df2.iloc[i][idx] == 1:
                _j = idx

        arr[_i,_j] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(arr.T)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len((df1.columns))))
    ax.set_yticks(np.arange(len((df2.columns))))
    # ... and label them with the respective list entries
    ax.set_xticklabels(df1.columns)
    ax.set_yticklabels(df2.columns)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(df1.columns)):
        for j in range(len(df1.columns)):
            text = ax.text(j, i, int(arr[j, i]),
                           ha="center", va="center", color="w", fontsize='small')

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()
