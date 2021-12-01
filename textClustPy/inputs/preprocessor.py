import re
import string
from nltk import word_tokenize, PorterStemmer
from nltk.util import everygrams
from nltk.corpus import stopwords


# This preprocessor is used to preprocess text data stemming from a stream in a way that was specified in the
# input config file.
class Preprocessor:
    '''
    The Preprocessor object is one of the three core components. It determines how incoming text obvservations should be preprocessed before clustering.
        
    :param language: Language that should be used for stopword identification. We use nltk for stopword detection.
    :type language: string
    :param stopword_removal: Boolean variable, indicating whether stopwords should be removed.
    :type stopword_removal: bool
    :param stemming: Boolean variable, indicating whether stopwords should be removed.
    :type stemming: bool
    :param punctuation: Boolean variable, indicating whether punctuation should be removed.
    :type punctuation: bool
    :param punctuation: Boolean variable, indicating whether hashtags should be removed (especially interesing for twitter input).
    :type punctuation: bool
    :param punctuation: Boolean variable, indicating whether usernames (@username) should be removed (especially interesing for twitter input).
    :type punctuation: bool
    :param punctuation: Boolean variable, indicating whether urls should be removed.
    :type punctuation: bool
    :param max_grams: What types of n-grams should be generated (max_grams=2 creates 1-grams as well as 2-grams). The higher max_grams the higher 
    :type punctuation: int
    '''

    config = dict()
    
    def __init__(self, language="english", stopword_removal=True, stemming=False, punctuation=True,
                 hashtag=True, username=True, url=True, max_grams=1, exclude_tokens = None):
        super().__init__()

        self.language = language
        self.punctuation = punctuation
        self.hashtag = hashtag
        self.username = username
        self.url = url
        self.max_grams = max_grams
        
        if exclude_tokens == None:
            self.exclude_tokens = []
        else:
            self.exclude_tokens = exclude_tokens

        if stopword_removal:
            self.stopwords = stopwords.words(self.language)
        else:
            self.stopwords = None

        if stemming:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None

    # Preprocess the data. Lemmatization, tokenization and so on
    def preprocess(self, observation_text):
        '''
        Preprocesses a string with the given settings     
        :param observation_text: input string
        :type observation_text: string
        '''
        text = observation_text
        

        # convert text to lower-case
        text = text.lower()
 
        # remove URLs
        if self.url:
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
            text = re.sub(r'http\S+', '', text)
        
        # remove usernames
        if self.username:
            text = re.sub('@[^\s]+', '', text)
        
        # remove the # in #hashtag
        if self.hashtag:
            text = re.sub(r'#([^\s]+)', r'\1', text)
        
        # Remove Punctuation
        if self.punctuation:
            ## remove multi exclamation mark
            text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
            # Initialize Punctuation set
            exclude = '’“' + string.punctuation
            # Check char characters to see if they are in punctuation
            text = [char for char in text if char not in exclude]
            # Join the characters again to form the string.
            text = ''.join(text)
        
        # remove exlude words
        if len(self.exclude_tokens) is not 0:
            resultwords  = [word for word in text.split() if word not in self.exclude_tokens]
            text = ' '.join(resultwords)

        ## here we create a frequency table of the current sentence
        processed_words = list()
        
        ## tokenize words
        words = word_tokenize(text)

        ## create ngrams
        n_grams = everygrams(word_tokenize(text), min_len=1, max_len=self.max_grams)
        words = [ '_'.join(grams) for grams in n_grams]

        for word in words:
            if word not in self.stopwords:
                if self.stemmer is not None:
                    word = self.stemmer.stem(word)
                processed_words.append(word)

        return processed_words

    

