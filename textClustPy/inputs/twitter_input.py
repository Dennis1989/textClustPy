import sys
import json

import tweepy
import string

from .inputs import Input
from .inputs import Observation

from abc import ABC, abstractmethod

## implementation of abstract class
class TwitterInput(Input):
    '''
    A twitter input accesses the twitter stream and directly applies textclust on the incoming data.
        
    :param api_key: Twitter API key
    :type api_key: string
    :param api_secret: Twitter API secret
    :type api_secret: string
    :param access_token: Twitter access token
    :type access_token: string
    :param access_secret: Twitter access secret
    :type access_secret: string
    :param terms: List of searchterms/hashtags that should be monitored in twitter
    :type terms: List of strings
    :param languages: Filter teweets by languages
    :type languages: List of strings
    :param callback: Callback function that expects one parameter of Tweepy type :class:`Status` (see http://docs.tweepy.org/en/latest/)
    :type callback: Function
    '''

    def __init__(self, api_key, api_secret, access_token, access_secret, terms, languages = ["en"],conf = None, callback= None, **kwargs):
        super().__init__(**kwargs)
        ## load config
        #self.conf = self.loadconfig("config.json")
        self.callback = callback
        api_key= self.conf["twitter"]["api-key"] if conf else api_key
        api_secret = self.conf["twitter"]["api-secret"]  if conf else api_secret
        access_token = self.conf["twitter"]["access-token"]  if conf else access_token
        access_secret = self.conf["twitter"]["access-secret"]  if conf else access_secret
        terms = self.conf["twitter"]["terms"]  if conf else terms
        languages = self.conf["twitter"]["languages"]  if conf else languages
        
        ## set credentials
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_secret)
        
        # Construct the API instance
        api = tweepy.API(auth)
        print("Starting Twitter Stream")
        myStreamListener = MyStreamListener(twitterinput = self)
        myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
        myStream.filter(track=terms, languages=languages)

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None, twitterinput = None):
        super().__init__(api = api)
        self.input = twitterinput

    def on_status(self, status):
        if self.input.callback is not None:
           self.input.callback(status)
        data = Observation(status.text, status.id_str, status.created_at, json.dumps(status._json))
        self.input.processdata(data)
        



