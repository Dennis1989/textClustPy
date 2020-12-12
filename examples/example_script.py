from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import CSVInput
from textClustPy.inputs.inputs import Observation
from textClustPy import microcluster
import time
import logging

def clust_callback(textclust):
        for item in textclust.get_microclusters().values():
            print(item.tf)


## create textclust instance
clust = textclust(config="textclust_config.json", callback = clust_callback)
preprocessor = Preprocessor(max_grams=2)

## create input
input = CSVInput(textclust=clust, preprocessor=preprocessor , config="input_config.json")

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
new_obs = Observation("testtest")
## get cluster assignment for new observation
assignment = clust.get_assignment([Observation("test"), 
    Observation("test number 2")], input, "micro")

## show closest micro cluster
clust.showmicrocluster(assignment[0], 10)
clust.showmicrocluster(assignment[1], 10)
