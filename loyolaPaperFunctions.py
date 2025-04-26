import numpy as np
from scipy.spatial.distance import mahalanobis
import itertools
import math

# columns are ranks and has a list of its row assignments, rows , cell locations specify which rows belong to that rank

# df: the pandas dataframe
# Return a dataframe of each 5 row chunk
def getRowChunks(df, startIndex, endIndex):
    chunk = df.iloc[startIndex:endIndex + 1]

    return chunk

# getDensityScore: used when there is a tie between cluster results for each projection
# clusterList: list containing (x,y,z...) (depends on dimension 5,10,15,etc.) coordinates of each data point
def getDensityScore(cluster, headerNames):
    '''
    dfDict = {}
    dfList = cluster
    dfNames = headerNames
    dfDict[dfNames] = dfList
    
    dataframe = pd.DatFrame(dfDict)
    '''
    # cluster[i] returns a list of your row
    # headerNames[i] returns the name of the attribute for the column
    
    cartesianProd = changeToList(list(itertools.product([i for i in range(len(cluster))],[i for i in range(len(cluster))])))
    cartesianProd = list(createReducedList(cartesianProd))
    print("cartesianProd = ", cartesianProd)
    dfDict = {}
    sumDistance = 0
    for tup in cartesianProd:
        p1 = cluster[tup[0]] # return list of n-dimensional points
        p2 = cluster[tup[1]]
        
        for i in range(len(p1)):
            val1 = p1[i]
            val2 = p2[i]
            distance = math.pow(abs(val1 - val2), 2)
            sumDistance += distance
            
    return math.sqrt(sumDistance)
    
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

def cartesianProduct(numDim, numRanks, rank, inputList, elementList = 0) :
    if elementList != 0 :
        cartesianProduct = changeToList(list(itertools.product(inputList,elementList)))
        #print(cartesianProduct)
        reducedList = list(createReducedList(cartesianProduct))
        combinationList = [ i.tolist() for i in np.array_split(reducedList,numRanks)  ] 
        evenDistribution(combinationList)
        return combinationList[rank]
    else :
        cartesianProduct = inputList.copy()
        for step in range(numDim-1) :
            cartesianProduct = changeToList(list(itertools.product(cartesianProduct,inputList)))
        reducedList = list(createReducedList(cartesianProduct))
        #print(reducedList)
        combinationList = [ i.tolist() for i in np.array_split(reducedList,numRanks)  ] 
        evenDistribution(combinationList)
        return combinationList[rank]
    
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

 
headerNames = ["a","b","c"]
cluster1 = [[1,2,3],[3,2,1],[1,2,2],[2,3,1],[1,1,3]]

print(getDensityScore(cluster1, headerNames))