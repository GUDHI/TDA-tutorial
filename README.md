
# Tutorial for Topological Data Analysis with the Gudhi Library

Topological Data Analysis (TDA) is a recent and fast growing  field providing a set of new topological and geometric tools to infer relevant features for possibly complex data. Here we propose a set of notebooks for the practice of TDA with the python Gudhi library together with popular machine learning and data sciences libraries.
See for instance [this paper](https://arxiv.org/abs/1710.04019) for an introduction to TDA for data sciences.

### 0 - Install Python Gudhi Library  

See the [installation page](http://gudhi.gforge.inria.fr/python/latest/installation.html) or [conda install](https://anaconda.org/conda-forge/gudhi)

### 1 - Simplex trees and simpicial complexes

Starting from a point cloud in $R^d$ or in a general metric space, TDA typically aims at extracting topological signatures from the point cloud. By studying the topological of the point clouds, we actually mean studying the topology of unions of balls centered at the point cloud (offsets). However, non-discrete sets such as offsets, and also continuous mathematical shapes like curves, surfaces and more generally manifolds, cannot easily be encoded as finite discrete structures. [Simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex) are therefore used in computational geometry to approximate such shapes.

A simplicial complex is a set of [simplices](https://en.wikipedia.org/wiki/Simplex), they can be seen as higher dimensional generalization of graphs. They are mathematical objects that are both topological and combinatorial, a property making them particularly useful for TDA. Here is an exemple of simplicial complex :

![CSexemple](Images/Pers14.PNG)
 
A Filtration is a increasing sequence of sub-complexes of a simplicial complex $K$, it can be seen as ordering the simplices included in the complex. Indeed, simpicial complexes often come with a specific order, as for [Vietoris-Rips complexes](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex), [Cech complexes](https://en.wikipedia.org/wiki/%C4%8Cech_complex) and [alpha complexes](https://en.wikipedia.org/wiki/Alpha_shape#Alpha_complex). 


In Gudhi, filtered simplicial complexes are encoded through a data structure called simplex tree. The vertices are represented by integers, the edges by couple of integers etc.
![CSexemple](http://gudhi.gforge.inria.fr/python/latest/_images/Simplex_tree_representation.png)


We first propose a [notebook](Tuto-GUDHI-simplex-Trees.ipynb) for illustrating the use of simplex trees to represent simplicial complexes.

Next, a [notebook](Tuto-GUDHI-simplicial-complexes-from-data-points.ipynb) presents how to build Rips complexes and alpha complexes (represented as simplex trees) from data points in $ R^d $. 

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
