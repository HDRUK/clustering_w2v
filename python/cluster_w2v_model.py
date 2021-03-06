import argparse
import sys

import matplotlib.pyplot as plt
import helper_functions as hf
import io_helper
from gensim import models
import logging
import time
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd



def main(args):
    parser = argparse.ArgumentParser(description='clusters words and scores clusters')
    parser.add_argument("--inputModel",help="w2v model file",
                        type=str)
    parser.add_argument("--outputDir",help="Path to save results",type=str)
    parser.add_argument("--clusterCount",help="number of clusters to produce",type=int)
    parser.add_argument("--termList",help="terms to score cluster",nargs='+')

    args = parser.parse_args(args)

    rootPath =args.outputDir
    num_clusters = args.clusterCount
    n_init =100
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('workflow')

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(rootPath+'run.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("loading word2vec models")
    model = models.Word2Vec.load(args.inputModel)
    model.init_sims(replace=True)
    word_vectors = model.wv.syn0.astype(np.float16)

    word_centroid_map = performKmeansClustering( logger, model, n_init, num_clusters, word_vectors)
    clusWC = performClusterWordCounts(logger, word_centroid_map)

    termlist = args.termList
    #count the hits in each cluster
    df = performKeywordClusterCounts(logger, word_centroid_map,termlist)
    #score clusters according to the algorithm described in the paper
    result_df = performClusterAnalysis(clusWC, df)
    result_df.to_csv(args.outputDir +'/results.csv')
    outputAllClusters(num_clusters, rootPath, word_centroid_map)

    plotResults(result_df,args.outputDir +'/plot.png')
    logger.info("job complete!")


def outputAllClusters(num_clusters, rootPath, word_centroid_map):
    for ii in range(0, num_clusters):
        list = hf.getOtherClusterWordsFromList([ii], word_centroid_map)
        io_helper.writeList(list, rootPath + '/clus_{0}.txt'.format( ii), True)


def performClusterAnalysis(clusWC, df):
    #####
    #
    # analysis
    #
    ####
    # apply feature rescaling
    df_rescale = df.transpose(copy=True)
    df_rescale = (df_rescale - df_rescale.min()) / (df_rescale.max() - df_rescale.min())
    # create result_df with scores (sum rescaled hits)
    result_df = pd.DataFrame({'tot_hits': df.sum(), 'rescaled_hits': df_rescale.sum(axis=1), 'clus_wc': clusWC})
    # normalise scores by cluster wc
    result_df['score_norm'] = result_df['rescaled_hits'] / result_df['clus_wc']
    # log cluster wc for graphing
    result_df['clus_wc_log'] = result_df['clus_wc'].apply(np.log10)

    return result_df


def performKeywordClusterCounts(logger, word_centroid_map, termlist):

    logger.info("scoring keyword list against vocabulary")
    # get df hits
    df = hf.scoreClustersDetailedFast(termlist, word_centroid_map)
    return df


def performClusterWordCounts(logger, word_centroid_map):
    logger.info("counting total words in each cluster")
    clusWC = hf.countWordsInClusters(word_centroid_map)
    return clusWC


def performKmeansClustering( logger, model, n_init, num_clusters, word_vectors):
    # clustering
    logger.info("performing K means cluster predictions")
    logger.info("Creating: %s clusters.", num_clusters)
    start = time.time()
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=-1, n_init=n_init, precompute_distances=True)
    centroids = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.wv.index2word, centroids))
    end = time.time()
    elapsed = end - start
    logger.info("Time taken for K Means clustering: %s seconds.", elapsed)
    logger.info("KMeans parameters: %s", kmeans_clustering.get_params())
    return word_centroid_map

def plotResults(result_df,path):
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(result_df['clus_wc_log'], result_df['score_norm'], marker='o',
                c=result_df['score_norm'], cmap='RdYlGn', s=40)
    plt.axis(ymax=(max(result_df['score_norm']) *1.1),ymin=-0.0002 )

    #std dev lines and mean score
    mad6 = result_df['score_norm'].mad() * 6
    mad3 = result_df['score_norm'].mad() * 3
    median = result_df['score_norm'].median()

    plt.axhline(y=mad6, color='r')
    plt.axhline(y=mad3, color='b')
    plt.axhline(y=median, color='m')
    plt.xlabel('log10 cluster size')
    plt.ylabel('Relevance score')
    plt.annotate('Median', xy=(5,median))
    plt.annotate('6 MAD', xy=(5, mad6))
    plt.annotate('3 MAD', xy=(5, mad3))
    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv[1:])