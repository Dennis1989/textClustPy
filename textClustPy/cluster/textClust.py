from textClustPy.cluster.microcluster import microcluster
from textClustPy.cluster.distances import distances


# we use sklearn for traditional clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

import logging
import math
import operator
import gensim
import gensim.downloader as api
import numpy as np
import pandas as pd
import nltk
import statistics
import json


from copy import deepcopy
import threading

# set logging options
logger = logging.getLogger()
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# textclust class
class textclust:
    """
    This class implements the textClust clustering algorithm.

    :param radius: Distance threshold to merge two micro-clusters
    :type radius:  float (default=0.5)
    :param _lambda: Fading factor of micro-clusters
    :type _lambda: float
    :param tgap: Time between outlier removal (default=100)
    :type tgap: float
    :param verbose: Verbose mode (default=false)
    :type tgap: bool 
    :param termfading: Logical whether individual terms should also be faded (default=true)
    :type termfading: bool
    :param realtimefading: Logical whether Natural Time or Number of observations should be used for fading (default=true)
    :type realtimefading: bool 
    :param micro_distance: Distance metric used for clustering micro clusters (default ="tfidf_cosine_distance)
    :type micro_distance: string
    :param macro_distance: Distance metric used for clustering macro clusters (default="tfidf_cosine_distance)
    :type macro_distance: string
    :param model: Name of the Word Embedding Model that can be used for clustering (default=None)
    :type model: string
    :param num_macro: Number of macro clusters that should be identified during the reclustering phase (default = 3)
    :type num_macro: int 
    :param min_Weight: Minimum weight of micro clusters to be used for reclustering (default = 0)
    :type min_Weight: float 
    :param config: Path and filename of external configuration file (default = None)
    :type config: string
    :param callback: Callback function that should be called after tgap steps (default = None)
    :type callback: function, optional

    :ivar n: number of processed documents
    :ivar omega: omega is defined as the minimum weight of a cluster :math:`2^{(-\lambda * gap)}`
    """
    # initialize variables
    microclusters = None
    assignment = None
    clusterId = None

    # constructor with default specification
    def __init__(self, radius=0.3, _lambda=0.0005, tgap=100, verbose=None,
                 termfading=True, realtimefading=True, micro_distance="tfidf_cosine_distance",
                 macro_distance="tfidf_cosine_distance", model=None, idf=True,
                 num_macro=3, minWeight=0, config=None, embedding_verification=False, callback=None, 
                 auto_r = False, auto_merge = True, sigma= 1
                 ):
        if config is not None:
            self.conf = self.loadconfig(config)
            self.embedding_verification = self.conf["embedding_verification"]
            self.radius = self.conf["radius"]
            self._lambda = self.conf["lambda"]
            self.tgap = self.conf["tgap"]
            self.verbose = False
            self.termfading = self.conf["termfading"]
            self.idf = self.conf["idf"]
            self.micro_distance = self.conf["micro_distance"]
            self.macro_distance = self.conf["macro_distance"]
            self.model = self.conf["model"]
            self.numMacro = self.conf["num_macro"]
            self.realtimefading = self.conf["realtimefading"]
            self.minWeight = self.conf["minWeight"]
            self.auto_r = self.conf["auto_r"]
            self.auto_merge = self.conf["auto_merge"]
            self.callback = callback
            self.sigma = sigma

        # if no config is provided, load parameter setting
        else:
            self.conf = {}
            self.embedding_verification = self.conf["embedding_verification"] = embedding_verification
            self.radius = self.conf["radius"] = radius
            self._lambda = self.conf["lambda"] = _lambda
            self.tgap = self.conf["tgap"] = tgap
            self.verbose = verbose
            self.termfading = self.conf["termfading"] = termfading
            self.idf = self.conf["idf"] = idf
            self.micro_distance = self.conf["micro_distance"] = micro_distance
            self.macro_distance = self.conf["macro_distance"] = macro_distance
            self.model = model
            self.conf["model"] = str(model)
            self.numMacro = self.conf["num_macro"] = num_macro
            self.realtimefading = self.conf["realtimefading"] = realtimefading
            self.minWeight = self.conf["minWeight"] = minWeight
            self.auto_r = auto_r
            self.auto_merge = auto_merge
            self.callback = callback
            self.sigma = sigma

        # Initialize important values
        #self.callback = None
        self.num_merged_obs = 0
        self.t = None
        self.lastCleanup = 0
        self.avgweight = 0
        self.n = 1
        self.omega = 2**(-1*self._lambda * self.tgap)

        self.assignment = dict()
        self.microclusters = dict()
        self.clusterId = 0  # Could also use n or t
        self.microToMacro = None
        self.upToDate = False
        self.realtime = None
        self.distsum = []
        self.dist_mean = 0
        self.outlier_dist_square = 0
        # log settings
        logger.info("---------------------------------------------------------")
        logger.info("Starting textclust with the following configuration: ")
        logger.info("radius: {}".format(self.radius))
        logger.info("_lambda: {}".format(self._lambda))
        logger.info("tgap: {}".format(self.tgap))
        logger.info("verbose: {}".format(self.verbose))
        logger.info("termfading: {}".format(self.termfading))
        logger.info("idf: {}".format(self.idf))
        logger.info("mi‚cro_distance: {}".format(self.micro_distance))
        logger.info("macro_distance: {}".format(self.macro_distance))
        logger.info("model: {}".format(self.model))
        logger.info("numMacro: {}".format(self.numMacro))
        logger.info("realtimefading: {}".format(self.realtimefading))
        logger.info("omega: {}".format(self.omega))
        logger.info("minWeight: {}".format(self.minWeight))
        logger.info("auto_r: {}".format(self.auto_r))
        logger.info("auto_merge: {}".format(self.auto_merge))
        logger.info("---------------------------------------------------------")

        # if word embeddings are used, models have to be initialized
        if(self.model is not None and isinstance(self.model, str)):
            logger.info("loading pre-trained word embedding model")
            self.model = api.load(self.model)
            logger.info("normalize model")
            self.model.init_sims(replace=True)
            logger.info("model normalized")

        # create a new distance instance for micro and macro distances.
        # from now the correct distance measure, specified in the config is used
        self.micro_distance = distances(self.micro_distance, self.model)
        self.macro_distance = distances(self.macro_distance, self.model)
        logger.info("distance metrics loaded")
        logger.info("---------------------------------------------------------")

    # load config file
    def loadconfig(self, filename):
        """
        Loads the config file

        :param filename: Relative name/path of the config file
        :type filename:  string
        """
        with open(filename) as json_file:
            return json.load(json_file)

    # delete predefined model
    def deleteModel(self):
        """
        Deletes the embedding model currently used
        """
        del(self.model)
        logger.info("model deleted")

    # change distance metric during runtime
    def changedistance(self, type, metric_name):
        """
        Changes the distance metric used for micro/macro clustering

        :param type: Either "micro" or "macro"
        :type type:  string
        :param metric_name: Name of the new distance metric
        :type metric_name:  string
        """
        if(type == "micro"):
            self.micro_distance = distances(metric_name, self.model)
        else:
            self.macro_distance = distances(metric_name, self.model)

    # create term frequency table
    def create_frequency_tables(self, preprocessed_words, textid):
        tf = {}
        for word in preprocessed_words:
            if word in tf:
                tf[word]["tf"] += 1
                tf[word]["ids"].append(textid)
            else:
                tf[word] = {}
                tf[word]["tf"] = 1
                tf[word]["ids"] = [textid]
        return tf

    # update clustering
    def update(self, text, id, time, realtime=None):
        """
        Updates the micro-clustering by incorporating a new observation
        :param text: A new text document that should be clustered
        :type text:  string
        :param id: Unique document id
        :type id:  int/double
        param time: Timestamp of the new text document. If realtimefading is enabled, this parameter has to be provided.
        :type time:  time, optional
        """

        # first we create a tfidf table from the input text
        tf = self.create_frequency_tables(text, id)

        # set up to date variable. it is set when everything is faded
        self.upToDate = False

        # check if realtime fading is on or not. specify current time accordingly
        if self.realtimefading:
            self.t = time
        else:
            self.t = self.n

        # realtime is only the current time non decoded to store for the plotter
        if realtime is not None:
            self.realtime = realtime

        clusterId = None

        # if there is something to process
        if len(tf) > 0:

            # create artificial micro cluster with one observation
            mc = microcluster(tf, self.t, 1, self.realtime, id, self.clusterId)

            # if idf is required, we calculate it from all micro clusters
            idf = None
            if self.idf == True:
                idf = self.calculateIDF(self.microclusters.values())

            # set minimum distance to 1
            min_dist = 1
            smallest_key = None
            
            sumdist= 0
            squaresum = 0
            counter = 0
            # calculate distances and choose the smallest one
            for key in self.microclusters.keys():

                dist = self.micro_distance.dist(
                    mc, self.microclusters[key], idf)
                
                ## only of the distance is somehow term related

                counter     = counter + 1
                sumdist    += dist
                squaresum  += dist**2

                ## store minimum distance and smallest key
                if dist < min_dist:
                    min_dist     = dist
                    smallest_key = key

            
            if self.auto_r:   
                ## if we at least have two close micro clusters
                if counter > 1:
                    
                    ## our threshold
                    mu = (sumdist-min_dist)/(counter-1) 
                    treshold = mu - self.sigma * math.sqrt(squaresum/(counter-1)  - mu**2)
                    #treshold = mu
                    if min_dist < treshold:
                        clusterId = smallest_key
            else:
                if min_dist < self.radius:
                        clusterId = smallest_key


            # if embedding verification is on, we test closest distance according to embedding method
            if self.embedding_verification is True and smallest_key is not None and clusterId is None and self.n > 1000:
                
                 ## cluster Id actually represents the number of created clusters
                mu = self.outlier_dist_mean/self.clusterId

         
                sigma = 1 * math.sqrt((self.outlier_dist_square/self.clusterId)- mu**2)
                #print(sigma)
                #threshold = mu  - sigma
                threshold = mu
                #print(threshold)
           
                if min_dist < threshold:
                    embedding_dist = self.macro_distance.dist(
                        mc, self.microclusters[smallest_key], idf)
                    
                                    
                    if embedding_dist <= threshold:
                        #print(list(mc.tf.keys()))
                        #print(list(self.microclusters[smallest_key].tf.keys()))
                        # print("---------------------")
                        clusterId = smallest_key
                        #print("NO")

                    #print("################################")
            # if we found a cluster that is close enough we merge our incoming data into it
            if clusterId is not None:
                #print("merge")
                self.num_merged_obs += 1
                ## add number of observations
                self.microclusters[clusterId].n += 1

                self.microclusters[clusterId].merge(
                    mc, self.t, self.omega, self._lambda, self.termfading, self.realtime)
                self.assignment[self.n] = clusterId

                #if self.embedding_verification:
                self.dist_mean += min_dist
                #print(self.dist_mean)
                #self.outlier_dist_square += min_dist*min_dist 

            # if no close cluster is found we create a new one
            else:
                clusterId = self.clusterId
                self.assignment[self.n] = clusterId
                self.microclusters[clusterId] = mc
                self.clusterId += 1
                
        else:
            print("error")

        # cleanup every tgap
        if self.lastCleanup is None or self.t-self.lastCleanup >= self.tgap:
            self.cleanup()
            
            if self.callback is not None:
                
                ## we create a callback object with all the current micro clusters
                callbackobject = {"microclusters":self.microclusters, "assignment": self.assignment, 
                                    "radius": self.radius, "time": self.t , "n":self.n}
                copy = deepcopy(callbackobject)
                th = threading.Thread(target=self.callback, args=[copy])
                th.start()



        self.n += 1
        return clusterId

    # calculate a distance matrix from all provided micro clusters
    def get_distance_matrix(self, clusters):

        # if we need IDF for our distance calculation, we calculate it from the micro clusters
        if self.idf:
            idf = self.calculateIDF(clusters.values())
        else:
            idf = None

        # get number of clusters
        numClusters = len(clusters)
        ids = list(clusters.keys())

        # initialize all distances to 0
        distances = pd.DataFrame(
            np.zeros((numClusters, numClusters)), columns=ids, index=ids)

        for idx, row in enumerate(ids):
            for col in ids[idx+1:]:
                # use the macro-distance metric to calculate the distances to different micro-clusters
                dist = self.macro_distance.dist(
                    clusters[row], clusters[col], idf)
                distances.loc[row, col] = dist
                distances.loc[col, row] = dist
        return distances

    # calculate idf based in all micro-clusters

    def calculateIDF(self, microclusters):
        result = {}
        for micro in microclusters:
            for k in list(micro.tf.keys()):
                if k not in result:
                    result[k] = 1
                else:
                    result[k] += 1
        for k in list(result.keys()):
            result[k] = 1 + math.log(len(microclusters)/result[k])
        return result

    # update weights according to the fading factor
    def updateweights(self):
        for micro in self.microclusters.values():
            micro.fade(self.t, self.omega, self._lambda,
                       self.termfading, self.realtime)

        # delete micro clusters with a weight smaller omega
        for key in list(self.microclusters.keys()):
            if self.microclusters[key].weight <= self.omega or len(self.microclusters[key].tf) == 0:
                logger.debug("delte micro cluster")
                del self.microclusters[key]

    # cleanup procedure

    def cleanup(self):
        logger.debug("initialize cleanup")
        
       
        #self.outlier_dist_square = 0 

        # set last cleanup to now
        self.lastCleanup = self.t

        # update curren cluster weights
        self.updateweights()
  

        # set deltaweights
        for micro in self.microclusters.values():
            
            # here we compute delta weights
            micro.deltaweight = micro.weight - micro.oldweight
            micro.oldweight = micro.weight

        # if auto merge is enabled, close micro clusters are merged together
        if self.auto_merge:
            self.mergemicroclusters()

        ## reset merged observation
        self.dist_mean = 0 
        self.num_merged_obs = 0
               
    def mergemicroclusters(self):
        micro_keys = [*self.microclusters]


        idf = self.calculateIDF(self.microclusters.values())
        i = 0
        if self.auto_r:
            threshold = self.dist_mean / (self.num_merged_obs+1)
            print("threshold" + str(threshold))
        else:
            threshold = self.radius

        while i < len(self.microclusters):
            j = i+1
            while j < len(self.microclusters):
                m_dist = self.micro_distance.dist(self.microclusters[micro_keys[i]], self.microclusters[micro_keys[j]], idf)
                ## lets merge them
                #print(threshold)
                #print("MDIST_"+str(m_dist))
                if m_dist < threshold:
                    print("merge")
                    self.microclusters[micro_keys[i]].merge(
                    self.microclusters[micro_keys[j]], self.t, self.omega, self._lambda, self.termfading, self.realtime)
                    del(self.microclusters[micro_keys[j]])
                    del(micro_keys[j])
                else:
                    j = j+1 
            i = i+1
            
    # show the contents of a specific micro cluster
    def showmicrocluster(self, id, num):

        micro = self.microclusters[id]
        logger.info("-------------------------------------------")
        logger.info("Summary of microcluster: " + str(id))

        # sort micro cluster terms according to ther frequency
        indices = sorted(range(len([i["tf"] for i in micro.tf.values()])),
                         key=[i["tf"] for i in micro.tf.values()].__getitem__, reverse=True)

        # get representative for micro cluster
        representatives = [list(micro.tf.keys())[i]
                           for i in indices[0:min(len(micro.tf.keys()), num)]]
        logger.info(representatives)
        logger.info("-------------------------------------------")

    # show top micro/macro clusters (according to weight)
    def showclusters(self, topn, num, type="micro"):
        """
        Prints out the top micro/macro clusters (sorted after weight)
        :param topn: Number of top clusters to display
        :type topn: int
        :param num: Number of cluster representatives shown for each cluster 
        :type num int
        :param type: Type of cluster (micro or macro)
        :type type  string
        """

        # first clusters are sorted according to their respective weights
        if type == "micro":
            sortedmicro = sorted(self.getmicroclusters(
            ).values(), key=lambda x: x.weight, reverse=True)
        else:
            sortedmicro = sorted(self.getmacroclusters(
            ).values(), key=lambda x: x.weight, reverse=True)

        logger.info("-------------------------------------------")
        logger.info("Summary of " + type + " clusters:")

        for micro in sortedmicro[0:topn]:
            # print(micro.tf)
            logger.info("----")
            logger.info("micro cluster id " + str(micro.id))
            logger.info("micro cluster weight " + str(micro.weight))

            # get indices of top terms
            indices = sorted(range(len([i["tf"] for i in micro.tf.values()])),
                             key=[i["tf"] for i in micro.tf.values()].__getitem__, reverse=True)

            # get representative and weight for micro cluster (room for improvement here?)
            representatives = [(list(micro.tf.keys())[i], micro.tf[list(micro.tf.keys())[
                                i]]["tf"]) for i in indices[0:min(len(micro.tf.keys()), num)]]
            for rep in representatives:
                logger.info(
                    "weight: " + str(round(rep[1], 2))+"\t token: " + str(rep[0]).expandtabs(10))
            # logger.info(representatives)

        logger.info("-------------------------------------------")

    # get top n micro clusters
    def gettopmicrocluster(self, topn):
        return sorted(self.microclusters.values(), key=lambda x: x.weight, reverse=True)

    # get top ids of top micro clusters
    def gettopmicroclusterids(self, topn):
        sortedmc = sorted(self.microclusters.values(),
                          key=lambda x: x.weight, reverse=True)
        return [i.id for i in sortedmc][0:topn-1]

    # update macro clusters and the corresponding micro-to-macro assignments

    def updateMacroClusters(self):
        # check if something changed since last reclustering
        if not self.upToDate:

            # first update the weights
            self.updateweights()

            # filter for weight threshold and discard outlier or emerging micro clusters
            micros = {key: value for key, value in self.microclusters.items(
            ) if value.weight > self.minWeight}

            numClusters = min([self.numMacro, len(micros)])

            # AT THE MOMENT WE USE SPECTRAL CLUSTERING WHICH REQUIRES A SIMILARITY MATRIX!
            if(len(micros)) > 1:
                logger.debug("start macro clustering")

                #clusterer = SpectralClustering(
                #    assign_labels='discretize', n_clusters=numClusters, random_state=0, affinity="precomputed")
                clusterer = AgglomerativeClustering(n_clusters=numClusters, linkage="complete", affinity="precomputed")

                # shift to similarity matrix
                #if self.conf["macro_distance"] == "word_mover_distance":
                    # print("WMD")
                #    distm = 1./(1.+self.get_distance_matrix(micros))
                #else:
                    #print("NO WMD")
                #    distm = 1- self.get_distance_matrix(micros)
                    #distm = 0.5*((1-self.get_distance_matrix(micros))+1)
                #distm = np.exp(-self.get_distance_matrix(micros))
                distm = self.get_distance_matrix(micros)
                print(distm)
                logger.debug(distm)

                assigned_clusters = list(clusterer.fit(distm).labels_)
            else:
                assigned_clusters = [1]

            # build micro to macro cluster assignment based on key and clustering result
            ## "micro cluster with key x belongs to macro cluster i"
            self.microToMacro = {
                x: assigned_clusters[i] for i, x in enumerate(micros.keys())}

            self.upToDate = True

    # get micro cluster ids to macro cluster ids
    def get_microToMacro(self):
        self.updateMacroClusters()
        return self.microToMacro

    # return assignment dict {observation: microcluster}. If micro cluster does not exist anymore, it returns None at its position
    def get_microAssignment(self):
        return {key: value if value in self.get_microclusters() else None for key, value in self.assignment.items()}

    # Return assignment dict {observation: macrocluster}. If micro cluster does not exist anymore, it returns None at its position
    def get_macroAssignment(self):
        self.updateMacroClusters()
        return {key: self.microToMacro[value] if value in self.microToMacro else None for key, value in self.get_microAssignment().items()}

    # alias for backwards compatibility
    def getmicroclusters(self):
        return self.get_microclusters()

    def get_microclusters(self):
        """
        Returns a list of all current micro clusters

        :return: List of returned :class:`textClustPy.microcluster` objects
        """
        return self.microclusters

    def getmacroclusters(self):  # alias for backwards compatibility
        return self.get_macroclusters()

    # here we get macro cluster representatives by merging according to microToMacro assignments
    def get_macroclusters(self):
        """
        Returns a list of all current micro clusters

        :return: List of macro cluster dictionaries.
        """
        self.updateMacroClusters()
        numClusters = min([self.numMacro, len(self.microclusters)])

        # create empty clusters
        macros = {x: microcluster({}, self.t, 0, self.realtime, None, x)
                  for x in range(numClusters)}

        # merge micro clusters to macro clusters
        for key, value in self.microToMacro.items():
            macros[value].merge(self.microclusters[key], self.t,
                                self.omega, self._lambda, self.termfading, self.realtime)

        return macros

    # for a new observation(s) get the assignment to micro or macro clusters
    def get_assignment(self, points, input, type):

        self.updateweights()
        # assignment is an empty list
        assignment = list()
        idf = None

        # if idf is needed, we calculate it
        if self.idf == True:
            idf = self.calculateIDF(self.microclusters.values())

        # for all given points we do get the cluster assignment
        for point in points:

            # inputs are also processed by the preprocessor
            processed_text = input.preprocessor.preprocess(point.text)
            # term frequency table is created
            tf = self.create_frequency_tables(processed_text, point.id)

            # proceed, if the processed text is not empty
            if len(tf) > 0:
                # create temporary micro cluster
                mc = microcluster(tf, 1, 1, self.realtime, None, None)

                # initialize distances to infinity
                dist = float("inf")
                closest = None

                # identify the closest micro cluster using the predefined distance measure
                for key in self.microclusters.keys():
                    if self.microclusters[key].weight > self.minWeight:
                        cur_dist = self.micro_distance.dist(
                            mc, self.microclusters[key], idf)
                        if cur_dist < dist:
                            dist = cur_dist
                            closest = key

                # add assignment
                assignment.append(closest)
                #self.showmicrocluster(self.microclusters[closest].id,10)

            # if our tf is empty we append an "NA" assignment, indicating that this observation should be discarded
            else:
                assignment.append("NA")

        # if we have more than one real assignment
        if len(assignment) > 1:
            # based on type, we either return micro cluster assignments or macro cluster
            if(type == "micro"):
                return assignment
            else:
                microtomacro = self.get_microToMacro()
                return ["NA" if x == "NA" else microtomacro[x] for x in assignment]

        # in case we only have one assignment
        else:
            if(type == "micro"):
                return assignment[0]
            else:
                return self.get_microToMacro()[assignment[0]]
