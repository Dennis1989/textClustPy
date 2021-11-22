from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import TwitterInput

## callback function which is called every tgap timesteps
def clust_callback(textclust):
        textclust.showclusters(5, 10, "micro")


## callback function that is called for each and every new obsertvation. Can be used to store 
def save_callback(id, time, text, object):
        print(text)
        return


## create textclust instance
clust = textclust(callback = clust_callback, radius=0.5, _lambda=0.01, tgap=10, auto_r=False, model=None, 
        embedding_verification= True, macro_distance="embedding_cosine_distance")

## initialize preprocessor with 2-grams
preprocessor = Preprocessor(max_grams=2)

## create input
TwitterInput("###", "###", 
"###", "###",  
["hi"], ["en"], textclust=clust, preprocessor=preprocessor,callback=save_callback)
