from random import randrange
import random
import time
import math
import statistics
import numpy as np
from random import randint
import gensim


## tf container has tf value and the original textids
class tfcontainer:
    def __init__(self, tfvalue,ids):
        self.tfvalue = tfvalue
        self.ids = ids

class microcluster:
    """    
    Micro-clusters are statistic summaries of the data stream.

    :ivar id: cluster id
    :ivar tf: terms and frequencies of the micro cluster tokens
    :ivar weight: the current weight of the microcluster
    :ivar time: last time the microcluster was updated
    :ivar textids: All ids of documents that were assigned to the micro cluster
    
    """

    ## Initializer / Instance Attributes
    def __init__(self, tf, time, weight, realtime, textid, clusterid):
        self.id = clusterid
        self.weight = weight
        self.time=time
        self.tf = tf
        self.oldweight = 0
        self.deltaweight = 0
        self.realtime = realtime
        self.textids = [textid]
        self.n = 1
    
    ## fading micro cluster weights and also term weights, if activated
    def fade(self, tnow, omega, _lambda, termfading, realtime):
        self.weight = self.weight * pow(2,-_lambda*(tnow-self.time))
        if termfading:
            for k in list(self.tf.keys()):                
                self.tf[k]["tf"] = self.tf[k]["tf"] * pow(2,-_lambda*(tnow-self.time))
                if self.tf[k]["tf"] <= omega:
                    del self.tf[k]  
        self.time = tnow
        self.realtime = realtime

    ## merging two microclusters into one
    def merge(self, microcluster, t, omega, _lambda, termfading, realtime):
        
        ## add textids
        self.textids = self.textids + microcluster.textids
        
        self.realtime = realtime
        
        self.weight = self.weight + microcluster.weight

        self.fade(t,omega,_lambda,termfading, realtime)
        microcluster.fade(t,omega,_lambda,termfading, realtime)

        self.time = t
        # here we merge an existing mc wth the current mc. The tf values as well as the ids have to be transferred
        for k in list(microcluster.tf.keys()):
            if k in self.tf:
                self.tf[k]["tf"] += microcluster.tf[k]["tf"]
                self.tf[k]["ids"]=self.tf[k]["ids"]+list(microcluster.tf[k]["ids"])
            else:
                self.tf[k] = {}
                self.tf[k]["tf"] = microcluster.tf[k]["tf"]
                self.tf[k]["ids"] = list(microcluster.tf[k]["ids"])
