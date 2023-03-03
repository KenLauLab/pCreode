#from distutils.core import setup
from setuptools import setup

setup( name='pcreode',
       version='2.2.2',
       description='p-Creode algorithm for mapping state transitions',
       url='https://github.com/KenLauLab/pCreode',
       author='Chuck Herring',
       author_email='charles.a.herring@Vanderbilt.edu',
       packages=['pcreode'],
       install_requires=[
           'numpy>=1.11.0',
           'pandas>=0.17.1',
           'python-igraph<=0.9.11',
           'matplotlib',
           'sklearn',
           'jgraph',
           'jupyter',
           'nvr',
           'cairocffi',
       ],
     )

