================================================
Examples
================================================

textClust is designed that it works with just a few lines of code. Basically three components are required to run the algorithm:
    
    - a :class:`textClustPy.textclust` instance 
    - a :class:`textClustPy.Preprocessor` instance
    - a :class:`textClustPy.Input` instance

The textClust instance represents the algorithm and its configuration. Here typical hyperparameters such as the fading factor are specified. The preprocessor instance determines how incoming text observations are preprocessed before they are used by textClust. 
The input instance specifies what kind of input is used. The :class:`textClustPy.Input` instance receives a :class:`textClustPy.textclust` and a :class:`textClustPy.Preprocessor` instance.


CSV input
-----------------------
This is a simple example of textClust which uses a static csv file as input:

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import CSVInput
    from textClustPy import microcluster

    ## create textclust and preprocessor
    clust = textclust(config="textclust_config.json")
    preprocessor = Preprocessor(max_grams=2)

    ## create input
    input = CSVInput(clust, preprocessor)

    ## update the algorithm (1000 steps)
    input.update(1000)

    ## show micro clusters
    clust.showclusters(len(clust.get_microclusters()), 10, "micro")



Pandas data frame input
-----------------------
It is also  possible to use a pandas data frame as a streaming source. In the following example, we use textClust on the well-known NewsAggregator dataset (see https://archive.ics.uci.edu/ml/datasets/News+Aggregator).

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import InMemInput
    import requests, zipfile, io
    import pandas as pd
    from datetime import datetime

    ## get dataset, unzip and load into pandas
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open('newsCorpora.csv'), sep="\t", header=None)

    ## create datetime from timestamp
    df[7] = [datetime.fromtimestamp(x/1000) for x in df[7]]

    ## create textclust instance
    clust = textclust(radius=0.7, _lambda=0.005)
    preprocessor = Preprocessor(max_grams=2)

    ## create input
    input = InMemInput(textclust=clust, preprocessor=preprocessor, pdframe =df, 
        col_id = 0, col_time = 7, col_text = 1, timeprecision="hours")

    ## update the algorithm (1000 steps)
    input.update(1000)

    ## show top 10 micro clusters
    clust.showclusters(10, 10, "micro")



Twitter input
-----------------------
Lastly, we use the Twitter Input to directly monitor the Twitter Stream with textClust! Since it is a live stream, where data is constantly arriving, a callback function must be provided. This callback function is called each tgap timesteps with a deep copy of the current textClust state. It is up to the user to deal with the current state. In this example we simply pront the top 5 micro-clusters.

.. code-block:: python

    from textClustPy import textclust
    from textClustPy import Preprocessor
    from textClustPy import TwitterInput

    ## callback function which is called every tgap timesteps
    def clust_callback(textclust):
            textclust.showclusters(5, 10, "micro")

    ## create textclust instance
    clust = textclust(config="textclust_config.json", callback = clust_callback)
    preprocessor = Preprocessor(max_grams=2)

    ## Create Twitter Input. API tokens and secrets are denoted as $$$$$
    TwitterInput("$$$$$", "$$$$$", "$$$$$", "$$$$$", 
    ["trump"], ["en"], textclust=clust, preprocessor=preprocessor)




