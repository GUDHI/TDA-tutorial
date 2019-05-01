
# Tutorial for Topological Data Analysis with the Gudhi Library

Topological Data Analysis (TDA) is a recent and fast growing  field providing a set of new topological and geometric tools to infer relevant features for possibly complex data. Here we propose a set of notebooks for the practice of TDA with the python Gudhi library together with popular machine learning and data sciences libraries.


See for instance [this paper](https://arxiv.org/abs/1710.04019) for an introduction TDA for data sciences

### 0 - Install Python Gudhi Library  

See the [installation page](http://gudhi.gforge.inria.fr/python/latest/installation.html) or [conda install](https://anaconda.org/conda-forge/gudhi)

### 1 - Simplex trees and simpicial complexes

Simplicial complexes can be seen as higher dimensional generalization of graphs. They are mathematical objects that are both topological and combinatorial, a property making them particularly useful for TDA.

In Gudhi, (filtered) simplicial complexes are encoded through a data structure called simplex tree. This [notebook](Tuto-GUDHI-simplex-Trees.ipynb) illustrates the use of simplex tree to represent simplicial complexes.

The next [notebook](Tuto-GUDHI-simplicial-complexes-from-data-points.ipynb) presents how to build Rips complexes and alpha complexes (represented as simplex trees) from data points in $R^d$. 

It is also possible to define Rips complexes in general metric spaces from a matrix of pairwise distances : see this [notebook](Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb).

The last [notebook]() of this section is about cubical complexes, which are filtered complexes defined on grids.


2 - Persitence homology and metrics


3 - Alternative representations of persistence and linearization


4 - Statistical tools for persistence


5 - Machine learning and deep learning for persistence


6- Alternative filtrations and robust TDA


7- Topological Data Analysis for Time series


8- Cover complexes and the Mapper Algorithm 


9- TDA and dimension reduction
