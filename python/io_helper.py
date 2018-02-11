import os, pickle
from sklearn.externals import joblib
import helper_functions as hf


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def save_obj(obj, path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path ):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def writeCounts(x, path):
    with open(path, 'w') as f:
        for k, v in x.iteritems():
            f.write(str(k))
            f.write('\t')
            f.write(str(v))
            f.write('\n')

def writeList(x, path,writeNewLine):
    with open(path, 'w') as f:
        for v in x:
            f.write(str(v))
            if writeNewLine:
                f.write('\n')
def saveKmeansModel(idx,path):
    joblib.dump(idx, path)

def writeClusterScores(x, path):
    with open(path, 'w') as f:
        for k, v in x.iteritems():
            if v is not None:
                for k1, v1 in v.iteritems():
                    f.write(str(k))
                    f.write('\t')
                    f.write(str(k1))
                    f.write('\t')
                    f.write(str(v1))
                    f.write('\n')

def writeClusterDictionaryToFile(terms,path,clusterFindingFunction,wordMatchingFunction,word_centroid_map):
    with open(path,'w') as f:
        allDicts = {}
        for term in terms:
            oneDict = hf.getOtherWordsFromWord(term,clusterFindingFunction,word_centroid_map)
            for i in range(0, len(oneDict.values())):
                allDicts[oneDict.keys()[i]] = oneDict.values()[i]
        for i in range(0, len(allDicts.values())):
            for word in allDicts.values()[i]:
                f.write(str(allDicts.keys()[i]))
                f.write('\t')
                f.write(str(word))
                f.write('\t')
                for term in terms:
                    match = wordMatchingFunction(term, str(word))
                    if match is not None:
                        f.write(str(match)+"|")
                f.write('\n')