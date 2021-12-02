================================================
Examples
================================================

textClust is designed that the algorithm works with just a few lines of code. Basically, three components are required to run the algorithm:
    
    - a :class:`textClustPy.textclust` instance 
    - a :class:`textClustPy.Preprocessor` instance
    - a :class:`textClustPy.Input` instance

The textClust instance represents the algorithm and its configuration. Here typical hyperparameters such as the fading factor are specified. The preprocessor instance determines how incoming text observations are preprocessed before they are used by textClust. 
The input instance specifies what kind of input is used. The :class:`textClustPy.Input` instance receives a :class:`textClustPy.textclust` and a :class:`textClustPy.Preprocessor` instance.


CSV input
-----------------------
This is a simple demonstration of textClust, using a static CSV file as input. We first download the well-known NewsAggregator dataset (see https://archive.ics.uci.edu/ml/datasets/News+Aggregator) and locally save it as a CSV file. Then we create a textClust and CSVInput instance.

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import CSVInput
    from textClustPy.inputs.inputs import Observation
    from textClustPy import microcluster
    import pandas as pd
    import pandas as pd
    from datetime import datetime
    import requests, zipfile, io


    def clust_callback(object):
            print(object["microclusters"])

    ## first we get a csv file
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open('newsCorpora.csv'), sep="\t", header=None)

    ## create datetime from timestamp
    df[7] = [datetime.fromtimestamp(x/1000) for x in df[7]]

    ## save locally as csv file
    df.to_csv("newsCorpora.csv",index=False,sep="\t")


    ## create textclust instance
    clust = textclust( radius=0.5, _lambda=0.01, tgap=10, auto_r=True, callback = clust_callback)
    preprocessor = Preprocessor(max_grams=2)

    ## create csv input
    input = CSVInput('newsCorpora.csv', textclust=clust, preprocessor=preprocessor , delimiter="\t",col_text=1,col_id=0,col_time=7,timeformat="%Y-%m-%d %H:%M:%S.%f", timeprecision="minutes")

    ## update the algorithm (1000 steps)
    input.update(1000)

    ## show top 5 micro clusters
    clust.showclusters(len(clust.get_microclusters()), 10, "micro")

    ## get micro clusters
    micro = clust.get_microclusters()

    ## get macro clusters
    macro = clust.getmacroclusters()

    ## get assignment from observation to micro cluster
    clust.get_macroAssignment()

    ## get assignment from observation to macro cluster
    clust.get_macroAssignment()

    ## get assignment from micro to macro cluster
    print(clust.get_microToMacro()) 

    ## show top 10 macro clusters
    clust.showclusters(10,10,"macro")

    ## create a new Observation
    new_obs = Observation("titanfall is a new game")

    ## get cluster assignment for new observations
    assignment = clust.get_assignment([Observation("titanfall is a new game"), 
        Observation("russian pipelines")], input, "micro")

    ## show closest micro cluster for both observations
    clust.showmicrocluster(assignment[0], 10)
    clust.showmicrocluster(assignment[1], 10)




Pandas data frame input
-----------------------
It is also possible to use a pandas data frame as a streaming source. In the following example, we use textClust on the well-known NewsAggregator dataset (see https://archive.ics.uci.edu/ml/datasets/News+Aggregator).

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import InMemInput
    import requests, zipfile, io
    import pandas as pd
    from datetime import datetime
    import numpy as np

    ## get dataset, unzip and load into pandas
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))

    df = pd.read_csv(z.open('newsCorpora.csv'), sep="\t", header=None)

    ## create datetime from timestamp
    df[7] = [datetime.fromtimestamp(x/1000) for x in df[7]]

    ## create textclust instance
    clust = textclust(radius=0.5, _lambda=0.05, auto_r= True, tgap=5, num_macro=10)
    preprocessor = Preprocessor(max_grams=2)

    ## create input
    input = InMemInput(textclust=clust, preprocessor=preprocessor, pdframe =df, 
        col_id = 0, col_time = 7, col_text = 1, timeprecision="minutes")

    ## update the algorithm (1000 steps)
    input.update(10000)

    ## show top 10 micro clusters
    clust.showclusters(10, 10, "micro")
    clust.showclusters(10, 10, type="macro")




Twitter input
-----------------------
Lastly, we use the Twitter Input to monitor the Twitter Stream with textClust directly! A callback function must be provided since it is a live stream, where data is constantly arriving. This callback function is called each tgap timesteps returning an object that contains important textClust instance attributes.

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import TwitterInput

    ## callback function which is called every tgap timesteps
    def clust_callback(object):
        print(object["microclusters"])


    ## callback function that is called for each and every new observation. Can be used to store 
    def save_callback(id, time, text, object):
        print(text)
        return

    ## create textclust instance
    clust = textclust(callback = clust_callback, radius=0.5, _lambda=0.01, tgap=10, auto_r=True)

    ## initialize preprocessor with 2-grams
    preprocessor = Preprocessor(max_grams=2)

    ## create Twitter input. Searching for english Tweets containing "hi" 
    TwitterInput("###", "###", 
    "###", "###",  
    ["hi"], ["en"], textclust=clust, preprocessor=preprocessor,callback=save_callback)





