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
#from ..util.inkrementalskip import inkrementalskip


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
    Abstract class of input

    :param textclust: A textclust instance of :class:`textClustPy.textclust`
    :type textclust: :class:`textClustPy.textclust`
    :param preprocessor: Preprocessor instance
    :type preprocessor: :class:`textClustPy.Preprocessor`
    :param timeformat: Specifies how time is formatted. Described as strftime directives (see https://strftime.org)
    :type config: string
    :param timeprecision: If realtimefading is enabled, timeprecision specifies on which time unit the fading factor is applied (seconds/minutes/hours). Default = "seconds"
    :type timeprecision: string
    :param config: relative path/name of config file
    :type config: string
    """

    def __init__(self, textclust, preprocessor, timeformat="%Y-%m-%d %H:%M:%S", timeprecision="seconds", config=None):
        super().__init__()
        self.ps = PorterStemmer()

        # load config
        self.conf = self.loadconfig("input_config.json") if config else None

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

        # tokenize words and remove stopwords and additional stuff
        processed_text = self.preprocessor.preprocess(observation.text)

        # if we have a live embedding, the model has to be updated with the processed text
        # if isinstance(self.clust.model, inkrementalskip):
        #    self.clust.model.train(processed_text)

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

            self.clust.update(processed_text, observation.id,
                              cur_time, observation.time)

        else:
            self.clust.update(processed_text, observation.id,
                              None, observation.time)

        self.debug_output()

        self.counter += 1
