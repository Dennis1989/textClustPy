from random import randrange
import random
import logging
import time
import math
import statistics
import numpy as np
from random import randint
import gensim
from gensim import utils, matutils

## set logging options
logger = logging.getLogger()
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

## distance class to implement different micro/macro distance metrics
class distances:
    def __init__(self, type, model):
        self.type = type
        self.model = model

    ## generic method that is called for each distance
    def dist(self, m1, m2, idf):
        return getattr(self,self.type,lambda:'Invalid distance measure')(m1, m2 ,idf)

    ## return a true tfidf vector
    def tfidf(self, micro, idf):
        result = np.zeros(len(idf))

        index = 0 
        for word in idf.keys():
            if word in micro.tf.keys():
                result[index] = micro.tf[word]["tf"]*idf[word]
            index = index + 1
        return result

    ##calculate cosine similarity directly
    def tfidf_cosine_distance(self, mc, microcluster, idf):
        sum = 0
        tfidflen = 0
        microtfidflen=0  
        for k in list(mc.tf.keys()):
            if k in idf:
                if k in microcluster.tf:
                    sum  += ((mc.tf[k]["tf"] *  idf[k]) * (microcluster.tf[k]["tf"] * idf[k]))
                tfidflen += (mc.tf[k]["tf"] * idf[k] * mc.tf[k]["tf"] * idf[k])
        tfidflen = math.sqrt(tfidflen)
        for k in list(microcluster.tf.keys()):
            microtfidflen += (microcluster.tf[k]["tf"]*idf[k] * microcluster.tf[k]["tf"] *idf[k])
        microtfidflen = math.sqrt(microtfidflen)
        if tfidflen==0 or microtfidflen==0:
            return 1
        else: 
            return round((1-sum/(tfidflen*microtfidflen)), 10)
    
    ##DEPRECATED: calculate extended cosine similarity bases on tf-idf
    def extended_cosine_distance(self, mc, microcluster, idf):
        sum = 0
        tfidflen = 0
        microtfidflen=0  
        meantf = statistics.mean([i["tf"] for i in mc.tf.values()])
        meanmicrotf = statistics.mean([i["tf"] for i in microcluster.tf.values()])
        for k in list(mc.tf.keys()):
            if k in idf:
                if k in microcluster.tf:
                    sum  += ((mc.tf[k]["tf"] *  idf[k] - meantf) * (microcluster.tf[k]["tf"] * idf[k])-meanmicrotf)
                tfidflen += ((mc.tf[k]["tf"] * idf[k]- meantf) * (mc.tf[k]["tf"] * idf[k]-meantf))
        tfidflen = math.sqrt(tfidflen)
        for k in list(microcluster.tf.keys()):
            microtfidflen += ((microcluster.tf[k]["tf"]*idf[k]-meanmicrotf) * (microcluster.tf[k]["tf"] *idf[k]-meanmicrotf))
        microtfidflen = math.sqrt(microtfidflen)
        if tfidflen==0 or microtfidflen==0:
            return 1
        else: return round((1 - sum/(tfidflen*microtfidflen)), 10)

    
  
    ## word mover distance based on model word embeddings
    def word_mover_distance(self, mc,  microcluster, idf):

        l1= list()
        l2= list()

        ## lets filter out important terms
        l1 = self.filter_clusters(mc)[0]

        ## lets filter out important terms
        l2 = self.filter_clusters(microcluster)[0]

        result = self.model.wmdistance(l1, l2) 
        logger.debug("distances: {}".format(result))
        
        return result

    ##  cosine distance of average word vectors
    def embedding_cosine_distance(self, mc, microcluster, idf):
        l1= list()
        l2= list()
        w1 = list()
        w2 = list()

        ## lets filter out important terms
        filters = self.filter_clusters(mc)
        l1 = filters[0]
        w1 = filters[1]

        ## lets filter out important terms
        filters = self.filter_clusters(microcluster)
        l2 = filters[0]
        w2 = filters[1]

            
        ## if l1 or l2 is empty the distance is 1
        if not l1 or not l2:
            return 1
        ## else return distance
        else:
            dist = 1 - self.model.n_similarity_weighted(l1,l2,w1,w2)
            logger.debug("distances: {}".format(dist))
            return dist

    ## DEPRECATED
    def weighted_embedding_dist(self, tokens1, tokens2, weights1, weights2):
        v1 = [self.model[word] * weights1[index] for index, word in enumerate(tokens1)]
        v2 = [self.model[word] * weights2[index] for index, word in enumerate(tokens2)]
        return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
    

    ## helper function to filter out all relevant tokens that should be used
    def filter_clusters(self, cluster, max_num=10):
        ## list of terms
        l = list()
        ## list of weights
        w = list()
       
        ## first we filter all terms with small weights
        filtered = {}
        for key,value in cluster.tf.items():
            if value["tf"]>=1:
                filtered[key] = value

        keys = list(filtered.keys())

        ## filter out identified tokens
        data = [{"key":keys[x],"tf":filtered[keys[x]]["tf"]} for x in sorted(range(len(filtered)),
                    key=[i["tf"] for i in filtered.values() ].__getitem__, reverse=True)[0:max_num]]

        ## all distances from input
        for key in data:
            ## check if we use a gensim model or a incremental model
            if hasattr(self.model, 'wv') and not key["key"] in self.model.wv:
                continue
            l.append(key["key"])
            w.append(key["tf"])
        return [l,w]
