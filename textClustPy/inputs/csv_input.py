from abc import ABC, abstractmethod

import sys
import json

import string
import csv
import time

from .inputs import Input
from .inputs import Observation

## implementation of abstract class
class CSVInput(Input):
    """
    This class implements the a csv input

    :param csvfile: Relative path and filename of the csv document
    :type csvfile:  string
    :param delimiter: Delimiter that separates different columns
    :type delimiter: char
    :param quotechar: Character that is used for quotes
    :type quotechar: char
    :param newline: Character indicating a new line.
    :type newline: char
    :param col_id: Column index that contains the text id 
    :type col_id: int
    :param col_time: Column index that contains the time
    :type col_time: int
    :param col_text: Column index that contains the text
    :type col_text: int
    :param col_label: Column index that contains the true cluster belonging
    :type col_label: int 
    """


    def __init__(self, csvfile=None, delimiter="|", quotechar=";", newline= "\n", 
    col_id=1, col_time=1, col_text=2, col_label=3, **kwargs):
        
        super().__init__(**kwargs)
        
        csvfile = self.conf["csv"]["file"] if self.conf else csvfile
        delimiter = self.conf["csv"]["sep"]  if self.conf else delimiter
        quotechar = self.conf["csv"]["quote"]  if self.conf else quotechar
        newline = self.conf["csv"]["newline"]  if self.conf else newline
        
        self.col_id = self.conf["csv"]["col_id"] if self.conf else col_id
        self.col_time = self.conf["csv"]["col_time"] if self.conf else col_time
        self.col_text = self.conf["csv"]["col_text"] if self.conf else col_text
        self.col_label = self.conf["csv"]["col_label"] if self.conf else col_label
        
        
        ## preload file
        print("preload file")
        self.csvfile    =   open(csvfile, newline=newline, encoding="UTF-8") 
        self.reader     =   csv.reader(self.csvfile, delimiter=delimiter, quotechar=quotechar)
        
        ## skip directly the header 
        next(self.reader)

        

    def run(self):
        '''
        Update the textclust algorithm with the complete data in the data frame
        '''
        for row in self.reader:

            text    = row[self.col_text]
            _id     = row[self.col_id]
            time    = row[self.col_time]
            label   = row[self.col_label] or None

            data    = Observation(text, _id, time, label, json.dumps(row))
            self.processdata(data)
                
               
    
    def update(self, n):
        '''
        Update the textclust algorithm on new observations 
        
        :param n: Number of observations that should be used by textclust
        :type n: int 
        '''
        for i in range(0,n):
            row     = next(self.reader)
            text    = row[self.col_text]
            _id     = row[self.col_id]
            time    = row[self.col_time]
            label   = row[self.col_label] or None

            data = Observation(text, _id, time, label, json.dumps(row))

            self.processdata(data)


    def fetch_from_stream(self, n):
        data = []
        for i in range(0, n):
            row     = next(self.reader)
            text    = row[self.col_text]
            _id     = row[self.col_id]
            time    = row[self.col_time]
            label   = row[self.col_label] or None

            data.append(Observation(text, _id, time, label, json.dumps(row)))
        return data
#CSVInput()
