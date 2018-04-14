from gensim import models
import logging
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('helperFunctions')
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
import numpy as np
from nltk.stem import *
import re
import copy
from scipy.spatial.distance import cdist, pdist
import pandas as pd
import io_helper as ioh
from sklearn.externals import joblib
###############clustering functions
def doClustering(w2vPath,savePath):
    start = time.time() # Start time

    model = models.Word2Vec.load(w2vPath)
    word_vectors = model.syn0.astype(np.float16)
    num_clusters = 40
    print ( "Creating: " + str(num_clusters) +" clusters.")


    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    joblib.dump(idx, savePath)
    logger.info( "Time taken for K Means clustering: " + str(elapsed) + " seconds.")
    logger.info("complete")

def doMiniBatchClustering(model,num_clusters,batch_size):
    word_vectors = model.syn0.astype(np.float16)
    print ( "Creating: " + str(num_clusters) +" clusters.")
    start = time.time()  # Start time
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = MiniBatchKMeans( n_clusters = num_clusters,batch_size=batch_size )
    idx = kmeans_clustering.fit_predict( word_vectors )
    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    # joblib.dump(idx, savePath)
    logger.info( "Time taken for K Means clustering: %s seconds.",elapsed )
    logger.info("complete")
    return idx


########Cluster Finding Fucntions

def findClustersWithOnlyExactMatches(word, word_centroid_map):
    x = set()
    x.add(word_centroid_map.get(word))
    return x

def findClustersWithAllExactMatches(word, word_centroid_map):
    returnSet = set()
    wordIterSet = set(word_centroid_map.keys())
    for i in wordIterSet:
        if word in i:
            returnSet.add(word_centroid_map[i])
    return returnSet

def findClustersWithAllRegExMatches(word, word_centroid_map):
    regex = re.compile('^'+word+'|_'+word)
    returnSet = set()
    wordIterSet = set(word_centroid_map.keys())
    for i in wordIterSet:
        if re.search(regex,i) is not None:
            returnSet.add(word_centroid_map[i])
    return returnSet

def findClustersWithStemMatches(word, word_centroid_map):
    x = set()
    for i in range(0, len(word_centroid_map.values())):
        term = word_centroid_map.keys()[i]
        token = matchStems(word,term)
        if token is not None:
          x.add(word_centroid_map.values()[i])
    return x
###############################



##########################################IO Functions



######################################################



######## Word Matching Functions
def matchStems(inputWord, queryWord):
    stemmer = PorterStemmer()
    inputTokens = str(inputWord).split("_")
    queryTokens = str(queryWord).split("_")
    for inputToken in inputTokens:
        for queryToken in queryTokens:
            if (stemmer.stem(inputToken) == stemmer.stem(queryToken)):
                return inputWord

def matchRegexes(inputWord, queryWord):
        regex = re.compile('^' + inputWord + '|_' + inputWord)
        if re.search(regex, queryWord) is not None:
            return inputWord

#############################

####other generic functions
def getOtherClusterWordsFromList(clusterNumList,word_centroid_map):
    words = set()
    for num in clusterNumList:
        words.update(getOtherClusterWords(num,word_centroid_map)[:])
    return words


def getOtherClusterWords(clusterNum,word_centroid_map):
    cluster_map = {}
    [cluster_map.setdefault(v, []).append(k) for (k, v) in word_centroid_map.items()]
    return cluster_map.get(clusterNum)

def getOtherWordsFromWord(word,matchingFunction, word_centroid_map):
    #set of cluster ID's
    clusters = matchingFunction(word,word_centroid_map)
    words = {}
    for cluster in clusters:
        newWords = getOtherClusterWords(cluster,word_centroid_map)
        words[cluster] = newWords
    return words



def scoreClustersDetailed(terms,matchingFunction, word_centroid_map):
    termHitDict = dict.fromkeys(terms, 0)
    clusterNumbers = set(word_centroid_map.values())
    wordList = word_centroid_map.keys()
    logger.info(str(clusterNumbers) + " clusters found")
    # clusterHitDict = dict.fromkeys(clusterNumbers)
    clusterHitDict = {key: copy.deepcopy(termHitDict) for key in clusterNumbers}
    index = len(word_centroid_map.values())
    counter = 0
    for word in wordList:
        if counter % 1000 ==0:
            logger.info('%s of %s words processed',counter,index)
        for term in terms:
            if matchingFunction(term,word) is not None:
                cluster = word_centroid_map.get(word)
                # if clusterHitDict[cluster] is None:
                #     clusterHitDict[cluster] = copy.deepcopy(termHitDict)
                currenttDict = clusterHitDict[cluster]
                currenttDict[term] +=1
                logger.info('cluster %s has %s hits on term %s',cluster, currenttDict[term],term)
        counter +=1
    return clusterHitDict

def scoreClustersDetailedFast(terms, word_centroid_map):
    # regexes = {}
    # for i in terms:
    #     regexes[i] = re.compile('^' + i + '|_' + i)
    termHitDict = dict.fromkeys(terms, 0)
    clusterNumbers = set(word_centroid_map.values())
    wordList = word_centroid_map.keys()
    logger.info('%s clusters found',len(clusterNumbers))
    # clusterHitDict = dict.fromkeys(clusterNumbers)
    clusterHitDict = {key: copy.deepcopy(termHitDict) for key in clusterNumbers}
    df = pd.DataFrame.from_dict(clusterHitDict)
    index = len(word_centroid_map.values())
    counter = 0
    for word in wordList:
        if counter % 100000 ==0:
            logger.info('%s of %s words processed',counter,index)
        for term in terms:
            if term in word:
                cluster = word_centroid_map.get(word)
                df.ix[term,cluster] +=1
        counter +=1
    return df



def scoreClustersSimple(terms,matchingFunction, word_centroid_map):
    clusterNumbers = set(word_centroid_map.values())
    wordList = word_centroid_map.keys()
    logger.info("%s clusters found",len(clusterNumbers))
    clusterHitDict = dict.fromkeys(clusterNumbers,0)
    index = len(word_centroid_map.values())
    counter = 0
    for word in wordList:
        if counter % 100000 ==0:
            logger.info('%s of %s words processed',counter,index)
        for term in terms:
            if matchingFunction(term,word) is not None:
                cluster = word_centroid_map.get(word)
                clusterHitDict[cluster] +=1
        counter +=1
    return clusterHitDict

def countWordsInClusters(word_centroid_map):
    clusterIds = set(word_centroid_map.values())
    clusterList = list(word_centroid_map.values())
    mapOfCounts = {}
    for id in clusterIds:
        x = clusterList.count(id)
        mapOfCounts[id] = x
    s = pd.Series(mapOfCounts)
    return s







#cluster, wc,hits,hitrate
def getClusterWCAndHits(wcDict, hitsDict):
    keys = wcDict.keys()
    clusterNum = len(keys)
    returnArray = np.zeros((clusterNum,4))
    keys = wcDict.keys()
    for i in range(0,clusterNum):
        returnArray[i,0] = i
        returnArray[i, 1] = wcDict[i]
        returnArray[i, 2] = hitsDict[i]
        if hitsDict[i] != 0:
            returnArray[i, 3] = returnArray[i, 2] / returnArray[i, 1]
    return returnArray

def getClustersWithMaxTermCount(ranked_clusters,maxTermCount):
    size = (ranked_clusters.shape[0]-1)
    wc = 0
    clusters = []
    for i in range(size,-1,-1):
        wc += ranked_clusters.iloc[i]['clus_wc']
        if wc >= maxTermCount:
            break
        else:
            clusters.append(ranked_clusters.iloc[i].name)
    return clusters






###score clusters {clusterno[term[count]]} agg

######################


######some interface functions





########################## plot it






###selecting K
def findKBatch(word_vectors,maxClusters,skip,repetitions,batch_size,init_size,n_init,reassignment_ratio,init='k-means++'):
    returnDict = {}
    index = 0
    for n in range(2,maxClusters,skip):
        resultsList = []
        for m in range(0,repetitions):
            kmeans_clustering = MiniBatchKMeans(n_clusters=n, batch_size=batch_size,init='k-means++',
                                                    init_size=init_size,n_init=n_init,reassignment_ratio=reassignment_ratio)
            centroids = kmeans_clustering.fit(word_vectors)
            resultsList.insert(0,pdist(centroids.cluster_centers_).min())
        returnDict[n] = np.mean(resultsList)
        index +=1
        logger.info('%s of %s K selection clusters performed',index, len(range(2,maxClusters,skip)))
    return returnDict

def findKNormal(word_vectors,maxClusters,skip,n_init,path):
    returnDict = {}
    index = 0
    for n in range(2,maxClusters,skip):
        kmeans_clustering = KMeans(n_clusters=n,n_jobs=-1,n_init=n_init,precompute_distances=True)
        centroids = kmeans_clustering.fit(word_vectors)
        returnDict[n] = pdist(centroids.cluster_centers_).min()
        index +=1
        logger.info('%s of %s K selection clusters performed',index, len(range(2,maxClusters,skip)))
        ioh.save_obj(returnDict, path)
    return returnDict



