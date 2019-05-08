
# Tutorial for Topological Data Analysis with the Gudhi Library

Topological Data Analysis (TDA) is a recent and fast growing  field providing a set of new topological and geometric tools to infer relevant features for possibly complex data. Here we propose a set of notebooks for the practice of TDA with the python Gudhi library together with popular machine learning and data sciences libraries.
See for instance [this paper](https://arxiv.org/abs/1710.04019) for an introduction to TDA for data sciences.

The complete list of notebooks can also be found at the end of this page.



### 0 - Install Python Gudhi Library  

See the [installation page](http://gudhi.gforge.inria.fr/python/latest/installation.html) or if you have conda you can make a [conda install](https://anaconda.org/conda-forge/gudhi).

### 1 - Simplex trees and simpicial complexes

TDA typically aims at extracting topological signatures from a point cloud in $\mathbb R^d$ or in a general metric space. By studying the topology of a point cloud, we actually mean studying the topology of the unions of balls centered at the point cloud, also called offsets. However, non-discrete sets such as offsets, and also continuous mathematical shapes like curves, surfaces and more generally manifolds, cannot easily be encoded as finite discrete structures. [Simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex) are therefore used in computational geometry to approximate such shapes.

A simplicial complex is a set of [simplices](https://en.wikipedia.org/wiki/Simplex), they can be seen as higher dimensional generalization of graphs. They are mathematical objects that are both topological and combinatorial, a property making them particularly useful for TDA. Below is an exemple of simplicial complex:

![title](Images/Pers14.PNG)
 
A filtration is a increasing sequence of sub-complexes of a simplicial complex K, it can be seen as ordering the simplices included in the complex K. Indeed, simpicial complexes often come with a specific order, as for [Vietoris-Rips complexes](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex), [Cech complexes](https://en.wikipedia.org/wiki/%C4%8Cech_complex) and [alpha complexes](https://en.wikipedia.org/wiki/Alpha_shape#Alpha_complex). 

[Notebook: Simplex trees](Tuto-GUDHI-simplex-Trees.ipynb). In Gudhi, filtered simplicial complexes are encoded through a data structure called simplex tree. The vertices are represented by integers, the edges by couple of integers etc.
![CSexemple](http://gudhi.gforge.inria.fr/python/latest/_images/Simplex_tree_representation.png)


[Notebook: Vietoris-Rips complexes and alpha complexes from data points](Tuto-GUDHI-simplicial-complexes-from-data-points.ipyn). In practice the first step of the "TDA pipeline analysis" is to define a filtration of simplicial complexes for some data. This  notebook explains how to build Vietoris-Rips complexes and alpha complexes (represented as simplex trees) from data points in $R^d$,using the simplex tree data structure.

[Notebook: Rips and alpha complexes from pairwise distance](Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb). It is also possible to define Rips complexes in general metric spaces from a matrix of pairwise distance, see this . We also give in this last notebook a way to define alpha complexes from matrix of pairwise distances by first applying a [multidimensional scaling (MDS)](https://en.wikipedia.org/wiki/Multidimensional_scaling) transformation on the matrix.


TDA signatures can extracted from point clouds but in many cases in data sciences the question is to study the topology of the sublevel sets of a function. 

![title](Images/sublevf.png)

Above is an example for a function defined on a subset of $R$ but in general the function f is defined on a subset of $R^d$. 

[Notebook: cubical complexes](Tuto-GUDHI-cubical-complexes.ipynb).  One first approach for studying the topology of the sublevel sets of a function is to define a regular grid on $R^d$ and then to define a filtered complex based on this grid by taking as a filter the function f.
 


todo : filtration defined on the top of a cs using an extra function



### 2 - Persistence homology and metrics


### 3 - Alternative representations of persistence and linearization


### 4 - Statistical tools for persistence


### 5 - Machine learning and deep learning for persistence


### 6- Alternative filtrations and robust TDA


### 7- Topological Data Analysis for Time series


### 8- Cover complexes and the Mapper Algorithm 


### 9- TDA and dimension reduction



## Complete list of notebooks for TDA

[Simplex trees](Tuto-GUDHI-simplex-Trees.ipynb) 

[Vietoris-Rips complexes and alpha complexes from data points](Tuto-GUDHI-simplicial-complexes-from-data-points.ipynb)  

[Rips and alpha complexes from pairwise distance](Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb)

[Cubical complexes](Tuto-GUDHI-cubical-complexes.ipynb)

