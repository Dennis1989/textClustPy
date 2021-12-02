from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import InMemInput
import requests, zipfile, io
import pandas as pd
from datetime import datetime
import numpy as np

## get dataset, unzip and load into pandas
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))

df = pd.read_csv(z.open('newsCorpora.csv'), sep="\t", header=None)

## create datetime from timestamp
df[7] = [datetime.fromtimestamp(x/1000) for x in df[7]]


## create textclust instance
clust = textclust(radius=0.5, _lambda=0.05, auto_r= True, tgap=5, num_macro=10)
preprocessor = Preprocessor(max_grams=2)

## create input
input = InMemInput(textclust=clust, preprocessor=preprocessor, pdframe =df, 
    col_id = 0, col_time = 7, col_text = 1, timeprecision="minutes")

## update the algorithm (1000 steps)
input.update(10000)

## show top 10 micro clusters
clust.showclusters(10, 10, "micro")


clust.showclusters(10, 10, type="macro")

