import setuptools
from subprocess import CalledProcessError
import sys

__version__ = "0.0.2"

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="textClustPy",
    version="0.0.1",
    author="Dennis Assenmacher and Matthias Carnein",
    author_email="dennis.assenmacher@wi.uni-muenster.de",
    description="A python implementation of textclust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://wiwi-gitlab.uni-muenster.de/d_asse011/textclustpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['nltk', 'pandas', 'gensim', 'numpy','sklearn', "tweepy", "elasticsearch", "jsonpickle"],
)