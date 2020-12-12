================================================
Configuration
================================================
If you want to run textClust, you have two configuration options. You can either pass the configuration parameters directly to the class instances or you can use configuration files.

There are two configuration files. One for textclust and one for the input. In the following an example of both files are provided.

First, an example of a textClust configuration:

.. code-block:: JSON

    {
        "lambda":0.001,
        "termfading":true,
        "tgap": 10.0,
        "radius":0.7,
        "realtimefading":true,
        "micro_distance":"tfidf_cosine_distance",
        "macro_distance":"tfidf_cosine_distance",
        "model":null,
        "idf": true,
        "num_macro":5,
        "minWeight": 1,
        "embedding_verification":false
    }


The Input configuration requires all parameters that are needed for the specified input (in this case a Twitter input):

.. code-block:: JSON

    {
        "general": {
            "live": true,
            "language": "english",
            "timeformat": "%Y-%m-%d %H:%M:%S",
            "uniquename": "name",
            "timeprecision": "seconds"
        },
        "twitter": {
            "terms": ["trump"],
            "languages": ["en"],
            "api-key": "###",
            "api-secret": "###",
            "access-token": "###",
            "access-secret": "###"
        }
    }

