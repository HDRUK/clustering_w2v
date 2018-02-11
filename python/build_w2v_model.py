import os
import sys
import nltk
import gensim
import logging
import argparse



def main(args):
    """builds a w2v model from a directory of plaintext files
    note, only basic configuration is available via CLI. Fine tuning
    and more complex use cases should edit this class directly.
    Builds a model with trigrams, bigrams and unigrams
    """
    parser = argparse.ArgumentParser(description='build W2V models with Gensim')
    parser.add_argument("--inputDir",help="directory containing "
                                                    "plain text files for building "
                                                    "W2V model",
                        type=str)
    parser.add_argument("--outputDir",help="Path to save output model",type=str)
    parser.add_argument("--epochs",help="training epochs to run",type=int)
    args = parser.parse_args(args)


    # set up logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    gensimLogger = logging.getLogger('gensim')
    logger = logging.getLogger('modelBuilder')
    fh = logging.FileHandler(os.path.join(args.outputDir, 'run.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    gensimLogger.addHandler(fh)


    model = gensim.models.Word2Vec(iter=args.epochs, workers=8)

    logger.info('beginning ngram vocab build')

    unigramGenerator= DiskSentenceGenerator(args.inputDir, logger)
    bigramGenerator= DiskSentenceGenerator(args.inputDir, logger)
    # trigramGenerator= unigramGenerator.copy()

    bigramTransformer = gensim.models.Phrases(bigramGenerator)

    bigramTransformer.save(os.path.join(args.outputDir,'bigram_model'))



    model.build_vocab(bigramTransformer[unigramGenerator])
    model.save(os.path.join(args.outputDir,'w2v_model'))


    logger.info('ngram vocab finished!')




class DiskSentenceGenerator(object):
    """simple generator class to produce sentences from plain text
    files in a directory. If further preprocessing is required,
    this is the place to do it"""
    def __init__(self, path,logger):
        self.path = path
        self.counter = 0
        self.logger = logger
        self.sentenceSplitter = nltk.data.load('tokenizers/punkt/english.pickle')

    def __iter__(self):
        for filename in os.listdir(self.path):
            with open(os.path.join(self.path,filename)) as f:
                content = f.read()
                sentences = self.sentenceSplitter.tokenize(content)
                for sentence in sentences:
                    yield nltk.word_tokenize(str(sentence).lower())
            self.counter += 1
            if self.counter % 1000 == 0:
                self.logger.info(str(self.counter) + ' docs processed')


if __name__ == "__main__":
    main(sys.argv[1:])