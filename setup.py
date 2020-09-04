#from distutils.core import setup
from setuptools import setup

setup( name='pcreode',
       version='2.2.0',
       description='p-Creode algorithm for mapping state transitions',
       url='https://github.com/KenLauLab/pCreode',
       author='Chuck Herring',
       author_email='charles.a.herring@Vanderbilt.edu',
       packages=['pcreode'],
       install_requires=[
           'numpy>=1.11.0',
           'pandas>=0.17.1',
           'matplotlib',
           'sklearn',
           'jgraph',
           'python-igraph',
           'jupyter',
           'nvr',
           'cairocffi',
       ],
     )

