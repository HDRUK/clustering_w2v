#Knowledge discovery for Deep Phenotyping serious mental illness from Electronic Mental Health records


This repository is in support of my paper @

https://f1000research.com/articles/7-210/v1

and is intented to demonstrate the methodology.

Note, since the original paper makes use of text data from electronic medical records, it is not possible to demonstrate
this on the intended dataset. Instead, I use a toy example, using the text of Les Miserables as a dataset (courtesy of
Project Gutenberg)

##Disclaimer

This is research code! No warranty etc..


##Dependencies

gensim
pandas
sklearn
matplotlib
nltk


##Overview

To build a Word2Vec model,

```
python build_w2v_model.py --inputDir <path> --outputDir <path> --epochs 10
```

inputDir should be a directory containing raw text (I'm using data/input)


##Clustering and scoring

With an embedding model built, we can apply clustering algorithms to group together similar concepts. I use KMeans,
 as it scales well (to estimate K, I suggest calculating minimum centroid distance, although other techniques exists).
 For the sake of brevity, let's assume  k=30. In order to score these clusters, we need to provide some prior information
 about them. FOr instance, say we're interested in clusters  related to javert cossette valjean (ideally all three):


 ```
 --inputModel <path> --outputDir <path> --clusterCount 30 --termList javert cossette valjean
 ```

 this will produce a csv in the outputDir called 'results.csv' with the score each cluster has received in a column
 called 'score_norm'. A plot of the results will be available in the file 'plot.png'. Green dots represent clusters with
 string hits, red a weak (or no) hits.

 please see the code for more information



