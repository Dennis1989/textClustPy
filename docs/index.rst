.. textClustPy documentation master file, created by
   sphinx-quickstart on Sun Dec  6 20:53:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
textClustPy's documentation!
================================================
.. image:: textClust.png
  :width: 400

textClust is a stream clustering algorithm for textual data that can identify and track topics over time in a stream of texts. The algorithm uses a widely popular two-phase clustering approach where the stream is first summarised in real-time. The result is many small preliminary clusters in the stream called 'micro-clusters'. Our micro-clusters maintain enough information to update and efficiently calculate the cosine similarity between them over time, based on the TF-IDF vector of their texts. Upon request, the miro-clusters can be reclustered to generate the final result using any distance-based clustering algorithm such as hierarchical clustering. To keep the micro-clusters up-to-date, our algorithm applies a fading strategy where micro-clusters that are not updated regularly lose relevance and are eventually removed.

.. toctree::
    :maxdepth: 2
   
    install
    examples
    API
    inputs
    preprocessor
    configuration


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

