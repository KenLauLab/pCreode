# **p-Creode**

[![Build Status](https://travis-ci.org/KenLauLab/pCreode.svg?branch=master)](https://travis-ci.org/KenLauLab/pCreode)  [![codecov](https://codecov.io/gh/KenLauLab/pCreode/branch/master/graph/badge.svg)](https://codecov.io/gh/KenLauLab/pCreode)   [![PyPI version](https://badge.fury.io/py/pcreode.svg)](https://badge.fury.io/py/pcreode)
[![Downloads](https://pepy.tech/badge/pcreode)](https://pepy.tech/project/pcreode)

### Please let us know if you run into any issues or have any questions, through the issues page above or by email: bob.chen@vanderbilt.edu. 

The term creode was coined by C.H. Waddington, combining the Greek words for “necessary” and “path” to describe the cell state transitional trajectories that define cell fate specification. Our algorithm aims to identify consensus routes from relatively noisy single-cell data and thus we named this algorithm p- (putative) Creode. Conceptually, p-Creode determines the geometric shape of a collection of dense data points (i.e., a data cloud) in order to reveal the underlying structure of transitional routes. p-Creode is compatible with Python 2 (2.7) and Python 3.  

Tutorial files:

[Jupyter notebook](https://github.com/KenLauLab/pCreode/blob/master/notebooks/pCreode_tutorial.ipynb)

Example [Data file](https://github.com/KenLauLab/pCreode/blob/master/data/Myeloid_with_IDs.csv)

### Installation for Mac or Linux

There are three ways to install p-Creode with Mac/Linux operating systems.

1.
```python
git clone git://github.com/KenLauLab/pCreode
cd pCreode
sudo pip install .
```
With this install, the jupyter notebook tutorial and the example scRNA-seq myeloid data set can be accessed in the notebook and data directories on your machine.

2.
```python
sudo pip install pcreode
```
Then manually download the [juypter notebook](https://github.com/KenLauLab/pCreode/blob/master/notebooks/pCreode_tutorial.ipynb) and example [data file](https://github.com/KenLauLab/pCreode/blob/master/data/Myeloid_with_IDs.csv) from the folders above. Simply right click on the download button and select "Save link as...".

Homebrew and anaconda install courtesy of Dan Skelly.

3.
```python
brew install igraph
git clone git://github.com/KenLauLab/pCreode
cd pCreode
conda create -n pcreode python=2.7 python-dateutil=2.5.3 numpy pandas matplotlib python-igraph jupyter cython
source activate pcreode
pip install .
```

Note: For Mac users, Cairo may or may not be installed. If you have problems plotting, you need to install Cairo. Please follow the instructions in this [link](http://macappstore.org/cairo/).

*** Some Mac users have experienced issues installing igraph, [here](http://igraph.org/python/#pyinstallosx) is a link to instructions for a direct install on a Mac 

### Installation for Windows

The problem with p-Creode installation on a Windows machine is with the python-igraph package, where there seems to be a bug in the setup.  Hence, additional steps must be taken.
1.       Install Anaconda
2.       Download the user compile wheels of 2 packages (download the version as appropriate to your Python install) from this [link](http://www.lfd.uci.edu/~gohlke/pythonlibs/): 

    -       [pyCairo](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pycairo)
  
            For example, in Anaconda prompt, go to the directory where the whl is downloaded and:  
   
            pip install pycairo‑1.18.1‑cp37‑cp37m‑win_amd64.whl
            
    -       [python-igraph](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph)
    
            pip install python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl
    
3. Install pcreode from github      

```
git clone git://github.com/KenLauLab/pCreode
cd pCreode
pip install .
```

*** if you are having issues with igraph plotting of graphs please try:
```
conda install -c conda-forge python-igraph
```

You will still need to manually download the [jupyter notebook](https://github.com/KenLauLab/pCreode/blob/master/notebooks/pCreode_tutorial.ipynb) and example [data file](https://github.com/KenLauLab/pCreode/blob/master/data/Myeloid_with_IDs.csv) from the folders above to be able to run the tutorial.

## Tutorial

Once p-Creode is installed you can access the tutorial by command line (conda environment if using PC) with
```python
jupyter notebook
```
The downloaded p-Creode tutorial can be opened by using the jupyter interface to find the directory where it was saved. A brief introduction to jupyter notebook can be found [here](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook#gs.P04bH=8).
