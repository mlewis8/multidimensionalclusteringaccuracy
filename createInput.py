
import numpy as np
import pandas as pd
import random
import sys

def evenDistribution(distribution) :
    largestList = max(len(x) for x in distribution )
    #print(largestList)
    #print("Largest list is %d "%(largestList))
    
    for entryList in distribution :
        if len(entryList) < largestList :
            remainder = largestList - len(entryList)
            #print(remainder)
            for i in range(remainder) :
                entryList.append(-1)

# clusterScript calls : python ${execFile} ${homeDir} ${cols[$i]} ${rows[$i]} ${rankSizes[$i]}
homeDirectory = sys.argv[1]
numTotalColumns = int(sys.argv[2])
numRows = int(sys.argv[3])
expRow = int(sys.argv[4])
numProcessors = int(sys.argv[5])

print("Create INPUT",homeDirectory,numTotalColumns,numRows,numProcessors)

myList = list(np.arange(expRow))
random.shuffle(myList)
print(myList)

rowDistribution = [ i.tolist() for i in np.array_split(myList,numProcessors)  ] 
evenDistribution(rowDistribution)

columnDf = pd.DataFrame()
for i in range(numProcessors) :
    columnDf.insert(i,"Rank#%d"%i, rowDistribution[i],True)

inputDf = pd.DataFrame()
for i in range(numTotalColumns) :
    columnList = np.random.uniform(low=0.5, high=20.0, size=(numRows,))
    random.shuffle(columnList)
    inputDf.insert(i,"Col#%d"%i,columnList,True)
    

columnDf.to_csv("%s/RANKS_rows_%d_cols_%d.csv"%(homeDirectory,expRow,numProcessors))
inputDf.to_csv("%s/DATA_rows_%d_cols_%d_procs_%d.csv"%(homeDirectory,numRows,numTotalColumns,numProcessors))