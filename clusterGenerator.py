from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import math
import sys
import itertools
from scipy.spatial.distance import cdist

# mpiexec -n 4 python clusterGenerator.py
# DEVELOPERS CAITLIN LAMIREZ 
# DR. MICHAEL LEWIS ( FUNCTIONS -- EUCLIDEAN, WNNERSLISTMAP,REPARTITION, GENERATECLUSTERS,GETCOMBINATIONS)
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

def changeToList(inputList) :
    #print(f"first item {inputList[0]} ")
    if isinstance(inputList[0][0],int) :
        return inputList
    #print(type(inputList[0]))
    returnList =  []
    for subList in inputList :
        #print(f"Sublist {subList}")
        tmpList = []
        setObj = subList[0]
        for elem in setObj :
            tmpList.append(elem)
        tmpList.append(subList[1])
        returnList.append(tmpList) 
    return returnList

def createReducedList(cartesianProduct) :
    lookUpTable = {}
    for itemList in cartesianProduct :
        setList = set(itemList)
        # Not entering duplicate items to the lookup table
        if len(itemList) != len(setList) :
            continue
        lookUpTable[tuple(setList)] = None
    reducedList = lookUpTable.keys()
    return reducedList

def cartesianProduct(numDim, numRanks, rank, inputList) :
    cartesianProduct = inputList.copy()
    for step in range(numDim-1) :
        cartesianProduct = changeToList(list(itertools.product(cartesianProduct,inputList)))
    reducedList = list(createReducedList(cartesianProduct))
    #print(reducedList)
    combinationList = [ i.tolist() for i in np.array_split(reducedList,numRanks)  ] 
    evenDistribution(combinationList)
    return combinationList[rank]

def generateKGroups(k,dataFrame) :
    numRows = len(dataFrame)
    groupList =  [ i.tolist() for i in np.array_split(np.arange(numRows).tolist(),k)  ]   
    #print("---",len(groupList))
    
    dataFrames = []  
    for indexList in groupList:
        newDf = dataFrame.iloc[indexList]
        dataFrames.append(newDf)
    return(dataFrames)

def ifNestedList(lst):
    for item in lst:
        if not isinstance(item, type(list)):
            return False
    return True 

def combineAllGather(rankList):
    completeAllGatherDict = {}
    
    # Goes through each list returned from each rank
    for rankDict in rankList:
        for key, val in rankDict.items():
            # If a nested list - ex: {0: [[1,2,3]], flatten it to remove brackets
            if ifNestedList(val):
                # Flatten list until extra brackets are removed
                while ifNestedList(val):
                    val = [x for sublist in val for x in sublist]
                
                
                completeAllGatherDict.setdefault(key, []).append(val)
                
            # Otherwise, just append the list to the key
            else:
                for x in val:
                    completeAllGatherDict.setdefault(key, []).append(x)
    
    return completeAllGatherDict

def euclidean(point, data, funcId):
    # Point represents all the features for a row ( data object)
    # data is the dataFrame for the entire cluster
    centroid = np.mean(data, axis=0)
    attributes = data.axes[1]
    squaredList = []
    #print(f"centroid : {type(centroid)} point : {type(point)} data : {type(data)}")
    #print(f"Centroid {centroid}")
    for attribute in attributes:
        value = ((point[attribute]) - (centroid[attribute])) ** 2
        #if funcId != 3 :
        #    if type(value) == pd.core.series.Series :
        #        print(f"Series value {type(value)} - {funcId} \n ")
        #    else :
        #        print(f"float value {type(value)} -  {funcId} \n")
                

        squaredList.append(value)
    return sum(squaredList)

def distributedRepartition(startIndex,rank, pair, allGatherRankDict,rankDistribution,geneDf,comm : MPI.COMM_WORLD) :

    if rank == 0 :
        startTime = time.time()
    # input dataframe the distributed points for a rank
    distDict = {}
    #print(f"Rank {rank} dFrame {dFrame.index.tolist()}")
    for i in rankDistribution[rank]:
        print("DistributedRepartition",geneDf)
        newDf = geneDf.loc[i + startIndex]   # inputDataFrame.loc[i]
        
        for key in allGatherRankDict :
            k_clusters = allGatherRankDict[key]
                # Result returns the list of d
            min = [math.inf, -1]
            for index, dataFrame in enumerate(k_clusters):
                #print(f"dFrame item : {newDf} \n")
                distance = euclidean(newDf, dataFrame,1)
                
                #distance = cdist(newDf,dataFrame,metric='euclidean')
                if distance < min[0]:
                    min[0] = distance
                    min[1] = index
            # Assign to distances
            clusterIndex = min[1]
            newList = list(key).copy()
            newList.append(clusterIndex)
            distDict.setdefault(tuple(newList),[]).append(i)

    allGatheredDistDict = comm.allgather(distDict)
    allGatheredDistDict = combineAllGather(allGatheredDistDict)
    combinedCombinationDict = {}
    for key in allGatheredDistDict :
        combination = tuple(key[0:len(key)-1])
        currentList = allGatheredDistDict[key]
        newDf = geneDf.iloc[currentList]
        #time.sleep(.001)
        combinedCombinationDict.setdefault(combination,[]).append(newDf)
    k_clusters = []
    if pair != -1 :
        k_clusters = combinedCombinationDict[tuple(pair)]
    return k_clusters

def distributedErrorFromDistance(rank, k_clusters,preTotalDistance,comm : MPI.COMM_WORLD) :
    if rank == 0 :
        startTime = time.time()
    totalDistance = 0.0
    continueClusterProcessing = 0
    #print(f"-- Total distance {totalDistance}")
    if k_clusters != [] :
        for dataFrame in k_clusters:
            if len(dataFrame) > 0 :
                centroidDataframe = np.mean(dataFrame, axis=0)
                centroidDataframe = pd.DataFrame(centroidDataframe).transpose()
                #print(centroidDataframe)
                #print(type(centroidDataframe))

                result = cdist(dataFrame,centroidDataframe,metric='mahalanobis')
                #print(f"Rank {rank} Resultant {np.sum(result)} ")
                
                totalDistance = totalDistance + np.sum(result)

                #for i in partition:
                #    rowDataFrame = dataFrame.loc[i]
                #    distance = euclidean(rowDataFrame, dataFrame,3)
                #    totalDistance = totalDistance + distance
        #print(f"preTotalDistance {type(preTotalDistance)} totalDistance {type(totalDistance)}")
        if abs(preTotalDistance - totalDistance) > .0001 :
            continueClusterProcessing = 1
    
    rankList = comm.allgather(continueClusterProcessing)
    
    return [totalDistance,1 in rankList]

def generateClusters(startIndex,k,rankDistribution,dataFrame,geneDf,rank=0,pair=0,comm : MPI.COMM_WORLD = 0) :
    # rankDistribution - the list of ids for this rank
    # dataFrame - the two column list for the dataFrame
    # geneDf - the entire dataset, this is needed for translating ids
    # k_clusters - the k partitions of the dataFrame
    rankDict = {}
    k_clusters = generateKGroups(k,dataFrame)
    if pair != -1 :
        rankDict[tuple(pair)] = k_clusters
   

    #------------- INFORMATION FOR DISTRIBUTED SECTION -----------------
    #    1. ALL TO ALL ON DICTIONARY (pair,k_clusters), (-1,-1) when pair = -1 
    #    2. CREATE A LIST OF K_CLUSTERS , INCLUDING THE -1
    #    2. COMM_BARRIER
    #-------------------------------------------------------------------
 
    #if rank == 0 :
    # printDictionary(-1,rank,rankDict)

    allGatheredRankDict = comm.allgather(rankDict)
    allGatheredRankDict = combineAllGather(allGatheredRankDict)

    #print(f"Rank {rank} keys: {rankDict.keys()}")    
    #if rank == 0 :
    #print(f"ALL GATHER DICT : {rank} -  {list(allGatheredRankDict.keys())} - {list(rankDict.keys()) }  ")                            
    
    #------------- LOOP OVER THE SAME AMOUNT OF CLUSTERS FOR DISTRIBUTED SECTION HERE -----------------
    
    preTotalDistance = 0
    continueProcessingTotal = True
    count = 0
    while True :
        # Every processor wether they have a dataFrame or not, needs to be a part of this process
        k_clusters = distributedRepartition(startIndex,rank,pair,allGatheredRankDict,rankDistribution,geneDf,comm)

        [totalDistance,continueProcessingTotal] = distributedErrorFromDistance(rank, k_clusters,preTotalDistance,comm)
        
        if continueProcessingTotal == False:
            comm.barrier()
            break
        preTotalDistance = totalDistance
        comm.barrier()
    return k_clusters

# numRowsIndex and numColumnsIndex are passed to this program

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

k = 3
numRowsIndex = 900
numRowsIndex = 510
numColumnsIndex = 7
numProcessors = size

homeDirectory = sys.argv[1]
cols = int(sys.argv[2])
startIndex = int(sys.argv[3])
endIndex = int(sys.argv[4])
numDim = int(sys.argv[5])
numRows = int(sys.argv[6])
expRow = int(sys.argv[7])
#dfHeader = pd.read_csv('your_data.csv', nrows=0)
#headerNames = list(dfHeader.columns)
rowData = []
print(homeDirectory,cols,startIndex,endIndex,numDim,numRows)


if rank == 0:
    
    df = pd.read_csv("%s/input/DATA_rows_%d_cols_%d_procs_%d.csv"%(homeDirectory,numRows,cols,numProcessors)).iloc[startIndex:endIndex,0:numColumnsIndex]
    
    inputDf = pd.read_csv("%s/input/RANKS_rows_%d_cols_%d.csv"%(homeDirectory,expRow,numProcessors))
    rankDistribution = []

    for i in range(numProcessors) :
        rankDistribution.append(list(inputDf.iloc[:,i+1]))

    for entryList in rankDistribution :
        if -1 in entryList :
            entryList.remove(-1)

    geneSymbols = list(df.iloc[:,0])
    geneDf = df.drop(df.columns[0],axis=1)
    lineageNames = list(geneDf)

    maxList = [ series.max() for name, series in geneDf.items()]
    minList = [ series.min() for name, series in geneDf.items()]
    totalMin = np.min(minList)
    totalMax = np.max(maxList)
    for name,series in geneDf.items() :
        geneDf[name] = geneDf[name].apply(lambda x: (x - totalMin) / (totalMax - totalMin))
else:
    geneDf = None 
    geneSymbols = None 
    lineageNames = None
    rankDistribution = None

try:
    geneDf = comm.bcast(geneDf, root=0)  
    geneSymbols = comm.bcast(geneSymbols, root=0) 
    lineageNames = comm.bcast(lineageNames, root=0)
    rankDistribution = comm.bcast(rankDistribution, root=0)
    
except Exception as e:
    print(f"Error in comm.bcast at rank {rank}: {e}")

# Cluster the randomDataFrame

topClusterPercentage = .05 
kMatrix = {}
winnerListDictWithK = {}
partitionPercentage = 0
timesGenerateClusterCalled = 0 
colList = [i for i in range(cols)]
myList = cartesianProduct(numDim,numProcessors,rank,colList)


print(myList)

print(geneDf)

#currDistribution = [ i + startIndex for i in rankDistribution[rank] ]
#for i in range(cols) :
##    distrList = geneDf.iloc[rankDistribution[rank],i].tolist()
#    print(distrList)
#exit(0)

for pair in myList :
    if pair != -1:
        pairSize = len(pair)
        dfDict = {}
        distrDict = {}
        for i in range(pairSize) :
            dfList = geneDf.iloc[:,pair[i]].tolist()
            dfName = geneDf.columns[pair[i]] 
            dfDict[dfName] = dfList
            #print("---------SUBSPACE----------")
            print("Name: ",dfName)
            print("Total Column Data",len(dfList))
            print("Pair ",pair[i])
            print("Rank Distribution ",rankDistribution[rank])
            #print("---------SUBSPACE----------")
            print("Start index",startIndex)
            #currentDistribution = [ i + startIndex  for i in rankDistribution[rank] ] 
            #print("RankDistribution",rankDistribution[rank])
            #print("Current distribution",currentDistribution)
            distrList = geneDf.iloc[rankDistribution[rank],pair[i]].tolist()
            #print(distrList)
            distrDict[dfName] = distrList
        dataframe = pd.DataFrame(dfDict)
        distributionDf = pd.DataFrame(distrDict)
            
    else:
        distributionDf = pd.DataFrame()
        dataframe = pd.DataFrame()

    k_clusters = generateClusters(startIndex,k,rankDistribution,dataframe,geneDf,rank, pair,comm)