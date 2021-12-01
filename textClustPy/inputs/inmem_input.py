from abc import ABC, abstractmethod

import sys

import json
import pandas as pd

import string
import csv
import time


from textClustPy import Input
from textClustPy import Observation

## implementation of abstract class
class InMemInput(Input):
    '''
    :param pdframe: Pandas data frame that serves as stream input
    :type pdframe:  DataFrame
    :param col_id: Column index that contains the text id 
    :type col_id: int
    :param col_time: Column index that contains the time
    :type col_time: int
    :param col_text: Column index that contains the text
    :type col_text: int
    :param col_text: Column index that contains the true cluster belonging
    :type col_label: int 
    '''

    def __init__(self, pdframe, col_id=1, col_time=1, col_text=2, col_label=None, **kwargs):
        
        ## call super constructor
        super().__init__(**kwargs)

        self.col_id = self.conf["inmemory"]["col_id"] if self.conf else col_id
        self.col_time = self.conf["inmemory"]["col_time"] if self.conf else col_time
        self.col_text = self.conf["inmemory"]["col_text"] if self.conf else col_text
        self.col_label = self.conf["inmemory"]["col_label"] if self.conf else col_label

        ## instance variable
        self.data =  pdframe
    
        # create iterator 
        self.reader =  self.data.itertuples(index=False)

        
    
    def run(self):
        '''
        Update the textclust algorithm with the complete data in the data frame
        '''
        for row in self.reader:
            data = self.getObservation(row)
            self.processdata(data)
                
               
    def getObservation(self, row):
        
        text = row[self.col_text]

        _id =  row[self.col_id]

        time = row[self.col_time]

        if self.col_label is not None:
            label = row[self.col_label]
        else:
            label = None

        data = Observation(text, _id, time, label, None)

        return data

    
    def update(self, n):
        '''
        Update the textclust algorithm on new observations 
        
        :param n: Number of observations that should be used by textclust
        :type n: int 
        '''
        for i in range(0,n):
            row = next(self.reader)
            data = self.getObservation(row)
            self.processdata(data)


    def fetch_from_stream(self, n):
        data = []
        for i in range(0,n):
            row = next(self.reader)
            obs = self.getObservation(row)
            data.append(obs)
        return data

