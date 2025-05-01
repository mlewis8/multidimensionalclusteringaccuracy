from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import math
import sys
import itertools
from scipy.spatial.distance import cdist
import loyolaPaperFunctions as loyoladf

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
        #print("DistributedRepartition",geneDf)
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
        print(key)
        newDf = geneDf.iloc[currentList,list(key[0:len(key)-1])]
        #time.sleep(.001)
        combinedCombinationDict.setdefault(combination,[]).append(newDf)
    k_clusters = []
    if pair != -1 :
        k_clusters = combinedCombinationDict[tuple(pair)]
    return k_clusters

def repartition(completeKclusters,originalDataFrame) :
    indexList = [ i for i in originalDataFrame.index]
    newK_clusters = []
    clusterAssignments = {}
    for i in indexList :
        # Create a single point dataFrame
        singleDataFrame = originalDataFrame.iloc[i]
        min = [math.inf, -1]
        for index,clusterDataFrame in enumerate(completeKclusters) :
            distance = euclidean(singleDataFrame, clusterDataFrame,1)
            if distance < min[0] :
                min[0] = distance
                min[1] = index
        clusterAssignments.setdefault(index,[]).append(i)
    for key,clusterList in clusterAssignments.items() :
        newCluster = originalDataFrame.iloc(clusterList)
        newK_clusters.append(newCluster)
    return newK_clusters

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
    #for cluster in k_clusters :
    #    print(rank,cluster)
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

    print(f"Rank {rank} keys: {rankDict.keys()}")    
    #if rank == 0 :
    #print(f"ALL GATHER DICT : {rank} -  {list(allGatheredRankDict.keys())} - {list(rankDict.keys()) }  ")                            
    
    #------------- LOOP OVER THE SAME AMOUNT OF CLUSTERS FOR DISTRIBUTED SECTION HERE -----------------
    
    preTotalDistance = 0
    continueProcessingTotal = True
    count = 0
    while True :
        # Every processor wether they have a dataFrame or not, needs to be a part of this process
        k_clusters = distributedRepartition(startIndex,rank,pair,allGatheredRankDict,rankDistribution,geneDf,comm)
        if count == 0 :
            for cluster in k_clusters :
                print(cluster)
        count += 1
        [totalDistance,continueProcessingTotal] = distributedErrorFromDistance(rank, k_clusters,preTotalDistance,comm)
        
        if continueProcessingTotal == False:
            comm.barrier()
            break
        preTotalDistance = totalDistance
        comm.barrier()
    return k_clusters

# [(0, 2), 25.658096116831313, 1]
def getProjectionRow(rankingList) :
    tempMap = {}
    finalMap = {}
    for item in rankingList :
        for projectionId in range(cols) :
            if projectionId in item[0] :
                tempMap.setdefault(projectionId,[]).append(item)

    winnerList = [0 for i in range(cols) ]
    densityAccum = [0 for i in range(cols)]
    returnList = []
    for projectionId,items in tempMap.items() :
        for item in items :
            densityScore = item[1]
            index = item[2]
            winnerList[index] += 1
            densityAccum[index] += densityScore
        topValue = [0, []] # Top value and density score
        for index,item in enumerate(winnerList) :
            if item > topValue[0] :
                topValue[1] = [index]
                topValue[0] = item
            elif item == topValue[0] :
                topValue[1].append(index)
        finalList = topValue[1]
        if len(finalList) == 1 :
            returnList.append(finalList[0])
        else :
            topDensityVal = 0 
            winningIndex = -1
            for index in finalList :
                densityVal = densityAccum[index]
                if densityVal > topDensityVal :
                    winningIndex = index
                    topDensityVal = densityVal
            returnList.append(winningIndex)
    return returnList

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
#print(homeDirectory,cols,startIndex,endIndex,numDim,numRows)


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

    #maxList = [ series.max() for name, series in geneDf.items()]
    #minList = [ series.min() for name, series in geneDf.items()]
    #totalMin = np.min(minList)
    #totalMax = np.max(maxList)
    #for name,series in geneDf.items() :
    #    geneDf[name] = geneDf[name].apply(lambda x: (x - totalMin) / (totalMax - totalMin))
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


#print(myList)

#print(geneDf)

#currDistribution = [ i + startIndex for i in rankDistribution[rank] ]
#for i in range(cols) :
##    distrList = geneDf.iloc[rankDistribution[rank],i].tolist()
#    print(distrList)
#exit(0)
clusterProjections = {}
densityMap = {}
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
            #print("Name: ",dfName)
            #print("Total Column Data",len(dfList))
            #print("Pair ",pair[i])
            #print("Rank Distribution ",rankDistribution[rank])
            #print("---------SUBSPACE----------")
            #print("Start index",startIndex)
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

    #if len(dataframe) > 0 :
    #    print(dataframe)
    k_clusters = generateClusters(startIndex,k,rankDistribution,dataframe,geneDf,rank, pair,comm)

   
   
    
    #print(k_clusters)
    # Initilizing of subspace
    # densityMap (pair) : [ density score1, densityScore2, densityScore3]
    # UPDATE (pair) : [ [[clusterlist],densityScore, id = -1], [clusterlist],densityScore1, id = -1]]
    # clusterProjection (rowId) : [pairId1, pairId2,...]
    
    for index,cluster in enumerate(k_clusters):
        #print(cluster)
        densityScore = loyoladf.getDensityScore(cluster)
        densityMap.setdefault(tuple(pair),[]).append([[i for i in cluster.index],densityScore])

# Calculate the full multidimensional cluster


for key in densityMap :
    listData = densityMap[key]
    sortedList = sorted(listData, key = lambda x: x[1])
    densityMap[key] = sortedList
    #print(key,densityMap[key])

sortedClusterMap = {} 
sortedDensityMap = {}
for key in densityMap :
    listData = densityMap[key]
    for val in listData :
        sortedClusterMap.setdefault(key,[]).append(val[0])
        sortedDensityMap.setdefault(key,[]).append(val[1])
                
#print(sortedClusterMap)
#print(sortedDensityMap)
try:
    sortedDensityMap = comm.allgather(sortedDensityMap)
except Exception as e:
    print(f"Error in comm.allgather at rank {rank}: {e}")

combineDensity = combineAllGather(sortedDensityMap)
#print(combineDensity)

try:
    sortedClusterMap = comm.allgather(sortedClusterMap)
except Exception as e:
    print(f"Error in comm.allgather at rank {rank}: {e}")

combineCluster = combineAllGather(sortedClusterMap)
#print(combineCluster)
#print(len(dataframe))

rankingMap = {}
projectionMap = {}

if rank == 0 :
    for rowId in range(len(dataframe)) :
        for pair, clusterListings in combineCluster.items() :
            densityList = combineDensity[pair]
            for index,clusterList in enumerate(clusterListings) :
                if rowId in clusterList :
                    rankingMap.setdefault(rowId,[]).append([pair,densityList[index],index])

    for rowId, item in rankingMap.items() :
        #print([i[0],i[2]] for i in item)
        projectionMap.setdefault(rowId,[]).append(getProjectionRow(item))
        

print(projectionMap)



#    print(combineDict)
#    [1 : [270,12], 20 : [2]]
#    
#print(rank, clusterProjections)
#clusterProjectTwo = {}
#for key,valueList in clusterProjections.items() :
#    # [ 270 : [[(0,2),1], (0,1), (3,1)] , 20 ]
#    for projectionId in range(cols) :
#        # check colIndex within valueList
#        for subspace in valueList :
#            pairId = subspace[0]
#            clusterId = subspace[1]
            
#            if projectionId in pairId :
#                clusterProjectTwo[key] = { projectionId : [clusterId,pairId,densityMap[tuple(pairId)][clusterId]]}



#try:
#    clusterProjectTwo = comm.allgather(clusterProjectTwo)
#except Exception as e:
#    print(f"Error in comm.allgather at rank {rank}: {e}")

#combineDict = combineAllGather(clusterProjectTwo)

#if rank == 0 :
#    print(clusterProjectTwo)

# ALL GATHER



#       Projection  : [ cluster index for each subspace] 
# 270 :  1 : [ 1, (0,1), 45 ]

# All Gather 
#270 : 1 : [ [ 1, (0,1), 45 ],  [ 2, (3,1), 15 ] ]
#270 : [ 1 : [1]] # Because the majority 