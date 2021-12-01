from abc import ABC, abstractmethod

import sys
import json

import string
import csv
import time

from .inputs import Input
from .inputs import Observation

## implementation of abstract class
class JsonInput(Input):
    """
    This class implements the a json input

    :param  jsonfile: Relative path and filename of the json document
    :type csvfile:  string
    :param col_id: Field name that contains the text id 
    :type col_id: int
    :param col_time:Field name that contains the time
    :type col_time: int
    :param col_text: Field name that contains the text
    :type col_text: int
    :param col_label: Field name that contains the true cluster belonging
    :type col_label: int 
    """


    def __init__(self, jsonfile=None, delimiter="|", quotechar=";", newline= "\n", 
    col_id=1, col_time=1, col_text=2, col_label=3, **kwargs):
        
        super().__init__(**kwargs)
        
        jsonfile = self.conf["json"]["file"] if self.conf else jsonfile
        
        self.col_id = self.conf["json"]["col_id"] if self.conf else col_id
        self.col_time = self.conf["json"]["col_time"] if self.conf else col_time
        self.col_text = self.conf["json"]["col_text"] if self.conf else col_text
        self.col_label = self.conf["json"]["col_label"] if self.conf else col_label
        
        
        ## preload file
        print("preload file")
        with open(jsonfile) as f:
            self.jsonfile = json.load(f)
        
        ## create iterator
        self.reader = iter(self.jsonfile)

        

    def run(self):
        '''
        Update the textclust algorithm with the complete data in the data frame
        '''
        for obj in self.reader:

            text    = obj[self.col_text]
            _id     = obj[self.col_id]
            time    = obj[self.col_time]
            label   = obj[self.col_label] or None

            data    = Observation(text, _id, time, label, json.dumps(obj))
            self.processdata(data)
                
               
    
    def update(self, n):
        '''
        Update the textclust algorithm on new observations 
        
        :param n: Number of observations that should be used by textclust
        :type n: int 
        '''
        for i in range(0,n):
            obj     = next(self.reader)
            text    = obj[self.col_text]
            _id     = obj[self.col_id]
            time    = obj[self.col_time]
            label   = obj[self.col_label] or None

            data = Observation(text, _id, time, label, obj)

            self.processdata(data)


    def fetch_from_stream(self, n):
        data = []
        for i in range(0, n):
            obj    = next(self.reader)
            text    = obj[self.col_text]
            _id     = obj[self.col_id]
            time    = obj[self.col_time]
            label   = obj[self.col_label] or None

            data.append(Observation(text, _id, time, label, obj))
        return data
#CSVInput()
