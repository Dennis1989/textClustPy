from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import CSVInput
from textClustPy.inputs.inputs import Observation
from textClustPy import microcluster
import pandas as pd
import pandas as pd
from datetime import datetime
import requests, zipfile, io


def clust_callback(object):
        print(object["microclusters"])

## first we get a csv file
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open('newsCorpora.csv'), sep="\t", header=None)

## create datetime from timestamp
df[7] = [datetime.fromtimestamp(x/1000) for x in df[7]]

## save locally as csv file
df.to_csv("newsCorpora.csv",index=False,sep="\t")


## create textclust instance
clust = textclust( radius=0.5, _lambda=0.01, tgap=10, auto_r=True, callback = clust_callback)
preprocessor = Preprocessor(max_grams=2)

## create csv input
input = CSVInput('newsCorpora.csv', textclust=clust, preprocessor=preprocessor , delimiter="\t",col_text=1,col_id=0,col_time=7,timeformat="%Y-%m-%d %H:%M:%S.%f", timeprecision="minutes")

## update the algorithm (1000 steps)
input.update(1000)

## show top 5 micro clusters
clust.showclusters(len(clust.get_microclusters()), 10, "micro")

## get micro clusters
micro = clust.get_microclusters()

## get macro clusters
macro = clust.getmacroclusters()

## get assignment from observation to micro cluster
clust.get_macroAssignment()

## get assignment from observation to macro cluster
clust.get_macroAssignment()

## get assignment from micro to macro cluster
print(clust.get_microToMacro()) 

## show top 10 macro clusters
clust.showclusters(10,10,"macro")

## create a new Observation
new_obs = Observation("titanfall is a new game")

## get cluster assignment for new observations
assignment = clust.get_assignment([Observation("titanfall is a new game"), 
    Observation("russian pipelines")], input, "micro")

## show closest micro cluster for both observations
clust.showmicrocluster(assignment[0], 10)
clust.showmicrocluster(assignment[1], 10)
