# **p-Creode**

The term creode was coined by C.H. Waddington, combining the Greek words for “necessary” and “path” to describe the cell state transitional trajectories that define cell fate specification. Our algorithm aims to identify consensus routes from relatively noisy single-cell data and thus we named this algorithm p- (putative) Creode. Conceptually, p-Creode determines the geometric shape of a collection of dense data points (i.e., a data cloud) in order to reveal the underlying structure of transitional routes. 

### Installation for Mac or Linux

There are two ways to install p-Creode with Mac/Linux operating systems. p-Creode has been implemented using Python2.7 and can be installed the following ways"
```python
git clone git://github.com/herrinca/pCreode
cd pCreode
sudo pip install .
```

### Installation for Windows

The problem with p-Creode installation on a Windows machine is with the python-igraph package, where there seems to be a bug in the setup.  Hence, additional steps must be taken.
1.       `Install Anaconda version 2.7`
2.       Download the user compile wheels of 2 packages (download the 32 bit version as appropriate):
    -       Cairo (pycairo-1.13.2-cp27-cp27m-win_amd64.whl)
    -       python-igraph (python_igraph-0.7.1.post6-cp27-none-win_amd64.whl)
3.       `pip install pcreode`

You will still need to manually download the [juypter notebook]() and example [data file]() from the folders above to be able to run the tutorial. Simply right click on the download button and select "Save link as...".
