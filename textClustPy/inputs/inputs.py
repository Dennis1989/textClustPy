from abc import ABC, abstractmethod
import logging
import json
import jsonpickle

import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, PorterStemmer
import string

import time
from datetime import datetime
import re



logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Observation:
    def __init__(self, text, id=None, time=None, label=None, object=None):
        self.text = text
        self.id = id
        self.time = time
        self.label = label
        self.object = object


class Input(ABC):
    """
    Abstract input class

    :param textclust: A textclust instance of type :class:`textClustPy.textclust`
    :type textclust: :class:`textClustPy.textclust`
    :param preprocessor: Preprocessor instance of type :class:`textClustPy.textclust`
    :type preprocessor: :class:`textClustPy.Preprocessor`
    :param timeformat: Specifies the time format. Described as strftime directives (see https://strftime.org). Default is: *%Y-%m-%d %H:%M:%S*
    :type config: string
    :param timeprecision: If realtimefading is enabled, timeprecision specifies on which time unit the fading factor is applied (seconds/minutes/hours). Default = "seconds"
    :type timeprecision: string
    :param config: Relative path/name of config file
    :type config: string
    :param callback: Callback function that is called for each incoming observation. The callback function expects four parameters: *ID, time, text* and a *Observation* object. 
    :type callback: function
    """

    def __init__(self, textclust, preprocessor, timeformat="%Y-%m-%d %H:%M:%S", timeprecision="seconds", config=None, callback = None):
        super().__init__()
        self.ps = PorterStemmer()

        # load config
        self.conf = self.loadconfig("input_config.json") if config else None
        self.callback = callback
        self.stopWords = set(stopwords.words(preprocessor.language))

        # create textclust instance based on given config
        self.clust = textclust
        self.preprocessor = preprocessor

        self.counter = 1
        self.start_time = time.time()

        self.timeformat = self.conf["general"]["timeformat"] if self.conf else timeformat
        self.timeprecision = self.conf["general"]["timeprecision"] if self.conf else timeprecision

    # print debug output

    def debug_output(self):
        # every 100 timesteps a summary of number of mcs
        if self.counter % 100 == 0 and self.counter > 1:
            logger.info("------")
            logger.info(self.clust.realtime)
            logger.info("#processed input data: "+str(self.counter))
            logger.info("#microclusters: "+str(len(self.clust.microclusters)))
            logger.info("elapsed time: "+str(time.time()-self.start_time))
            self.start_time = time.time()

    # load config file
    def loadconfig(self, filename):
        with open(filename) as json_file:
            return json.load(json_file)

    
    # this is always the same. Data is processed
    def processdata(self, observation):
        
        ## if callback is set call it
        if self.callback is not None:
            self.callback(observation.id, observation.time, observation.text, observation.object)

        # tokenize words and remove stopwords and additional stuff
        processed_text = self.preprocessor.preprocess(observation.text)


        # now we handle realtime fading
        if self.clust.realtimefading:

            # if the time is alrady datetime
            if isinstance(observation.time, datetime):
                cur_time = observation.time.timestamp()

            # string has to be converted to datetime
            else:
                cur_time = datetime.strptime(
                    observation.time, self.timeformat).timestamp()

            # now we check on which time precision should be used. Standard is seconds (as is)
            if self.timeprecision == "minutes":
                cur_time = cur_time / 60
            if self.timeprecision == "hours":
                cur_time = cur_time / 60

            clustID = self.clust.update(processed_text, observation.id,
                              cur_time, observation.time)

        else:
            clustID = self.clust.update(processed_text, observation.id,
                              None, observation.time)

        self.debug_output()

        self.counter += 1
        
        return clustID


