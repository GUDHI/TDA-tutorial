# Tutorial for Topological Data Analysis with the Gudhi Library

Topological Data Analysis (TDA) is a recent and fast growing  field providing a set of new topological and geometric tools to infer relevant features for possibly complex data. Here we propose a set of notebooks for the practice of TDA with the python Gudhi library together with popular machine learning and data sciences libraries.
See for instance [this paper](https://arxiv.org/abs/1710.04019) for an introduction to TDA for data sciences.

The complete list of notebooks can also be found at the end of this page.

### 0 - Install Python Gudhi Library  

See the [installation page](http://gudhi.gforge.inria.fr/python/latest/installation.html) or if you have conda you can make a [conda install](https://anaconda.org/conda-forge/gudhi).

### 1 - Simplex trees and simpicial complexes

TDA typically aims at extracting topological signatures from a point cloud in $\mathbb R^d$ or in a general metric space. By studying the topology of a point cloud, we actually mean studying the topology of the unions of balls centered at the point cloud, also called offsets. However, non-discrete sets such as offsets, and also continuous mathematical shapes like curves, surfaces and more generally manifolds, cannot easily be encoded as finite discrete structures. [Simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex) are therefore used in computational geometry to approximate such shapes. 

A simplicial complex is a set of [simplices](https://en.wikipedia.org/wiki/Simplex), they can be seen as higher dimensional generalization of graphs. They are mathematical objects that are both topological and combinatorial, a property making them particularly useful for TDA. The challenge here is to define such structures that are proven to reflect relevant information about the structure of data and that can be effectively constructed and manipulated in practice. Below is an exemple of simplicial complex:

![title](Images/Pers14.PNG)
 
A filtration is a increasing sequence of sub-complexes of a simplicial complex K, it can be seen as ordering the simplices included in the complex K. Indeed, simpicial complexes often come with a specific order, as for [Vietoris-Rips complexes](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex), [Cech complexes](https://en.wikipedia.org/wiki/%C4%8Cech_complex) and [alpha complexes](https://en.wikipedia.org/wiki/Alpha_shape#Alpha_complex). 

[Notebook: Simplex trees](Tuto-GUDHI-simplex-Trees.ipynb). In Gudhi, filtered simplicial complexes are encoded through a data structure called simplex tree. The vertices are represented by integers, the edges by couple of integers etc.
![CSexemple](http://gudhi.gforge.inria.fr/python/latest/_images/Simplex_tree_representation.png)


[Notebook: Vietoris-Rips complexes and alpha complexes from data points](Tuto-GUDHI-simplicial-complexes-from-data-points.ipynb). In practice the first step of the "TDA pipeline analysis" is to define a filtration of simplicial complexes for some data. This  notebook explains how to build Vietoris-Rips complexes and alpha complexes (represented as simplex trees) from data points in $R^d$,using the simplex tree data structure.

[Notebook: Rips and alpha complexes from pairwise distance](Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb). It is also possible to define Rips complexes in general metric spaces from a matrix of pairwise distance. The definition of the metric on the data is usually given as an input or guided by the application. It is however important to notice that the choice of the metric may be critical to reveal interesting topological and geometric features of the data.We also give in this last notebook a way to define alpha complexes from matrix of pairwise distances by first applying a [multidimensional scaling (MDS)](https://en.wikipedia.org/wiki/Multidimensional_scaling) transformation on the matrix.


TDA signatures can extracted from point clouds but in many cases in data sciences the question is to study the topology of the sublevel sets of a function. 

![title](Images/sublevf.png)

Above is an example for a function defined on a subset of $R$ but in general the function f is defined on a subset of $R^d$. 

[Notebook: cubical complexes](Tuto-GUDHI-cubical-complexes.ipynb).  One first approach for studying the topology of the sublevel sets of a function is to define a regular grid on $R^d$ and then to define a filtered complex based on this grid and the function f.
 




### 2 - Persistence homology and persistence diagrams

Homology is a classical concept in algebraic topology providing a powerful tool to formalize and handle the notion of topological features of a topological space or of a simplicial complex in an algebraic way. For any dimension k, the k-dimensional "holes" are represented by a vector space Hk whose dimension is intuitively the number of such independent features. For example the 0-dimensional homology group H0 represents the connected components of the complex, the 1-dimensional homology group H1 represents the 1-dimensional loops, the 2-dimensional homology group H2 represents the 2-dimensional cavities...

Persistent homology is a powerful tool to compute, study and encode efficiently multiscale topological features of nested families of simplicial complexes and topological spaces. It encodes the evolution of the homology groups of the nested complexes across the scales.

Persistent homology is a powerful tool to compute, study and encode efficiently multiscale topological features of nested families of simplicial complexes and topological spaces. It encodes the evolution of the homology groups of the nested complexes across the scales. The diagram below shows several level sets of the filtration.

![title](Images/pers.png)    
    

[Notebook: persistence diagrams](Tuto-GUDHI-persistence-diagrams.ipynb) In this notebook we show how to compute barcodes and persistence diagram from a filtration defined on the Protein binding dataset. This tutorial also introduces the bottleneck distance between persistence diagrams. 


### 3 - Alternative representations of persistence and linearization


### 4 - Statistical tools for persistence
For many applications of persistent homology, we observe topological features close to the diagonal. Since they correspond to topological structures that die very soon after they appear in the filtration, these points are generally considered as "topological noise". Confidence regions for persistence diagram provide a rigorous framework to this idea. This [notebook](Tuto-GUDHI-ConfRegions-PersDiag-datapoints.ipynb) introduces the subsampling approach of [Fasy etal. 2014 AoS](https://projecteuclid.org/download/pdfview_1/euclid.aos/1413810729). An alternative method is the bottleneck bootstrap method introduced in [Chazal etal. 2018](http://www.jmlr.org/papers/v18/15-484.html) and presented in this [notebook](Tuto-GUDHI-ConfRegions-PersDiag-BottleneckBootstrap.ipynb).

### 5 - Machine learning and deep learning with TDA

Two libraries related to Gudhi:   
- [ATOL](https://github.com/martinroyer/atol): Automatic Topologically-Oriented Learning. See [this tutorial](https://github.com/martinroyer/atol/blob/master/demo/atol-demo.ipynb).     
- [Perslay](https://github.com/MathieuCarriere/perslay): A Simple and Versatile Neural Network Layer for Persistence Diagrams. See [this tutorial](https://github.com/MathieuCarriere/perslay/tree/master/tutorial).


### 6- Alternative filtrations and robust TDA

This  [notebook](Tuto-GUDHI-DTM-filtrations.ipynb) introduces the distance to measure (DTM) filtration, as defined in [this paper](https://arxiv.org/abs/1811.04757). This filtration can be used for robust TDA. The DTM can also be used for robust approximations of compact sets, see this [notebook](Tuto-GUDHI-kPDTM-kPLM.ipynb).


### 7- Topological Data Analysis for Time series


### 8- Cover complexes and the Mapper Algorithm 


### 9- TDA and dimension reduction



## Complete list of notebooks for TDA

[Simplex trees](Tuto-GUDHI-simplex-Trees.ipynb) 

[Vietoris-Rips complexes and alpha complexes from data points](Tuto-GUDHI-simplicial-complexes-from-data-points.ipynb)  

[Rips and alpha complexes from pairwise distance](Tuto-GUDHI-simplicial-complexes-from-distance-matrix.ipynb)

[Cubical complexes](Tuto-GUDHI-cubical-complexes.ipynb)

[Persistence diagrams and bottleneck distance](Tuto-GUDHI-persistence-diagrams.ipynb)

[Confidence regions for persistence diagrams - data points](Tuto-GUDHI-ConfRegions-PersDiag-datapoints.ipynb)

[Confidence regions with Bottleneck Bootstrap](Tuto-GUDHI-ConfRegions-PersDiag-BottleneckBootstrap.ipynb)

[ATOL tutorial](https://github.com/martinroyer/atol/blob/master/demo/atol-demo.ipynb)   

[Perslay tutorial](https://github.com/MathieuCarriere/perslay/tree/master/tutorial)

[DTM-filtrations](Tuto-GUDHI-DTM-filtrations.ipynb) 

[kPDTM-kPLM](Tuto-GUDHI-kPDTM-kPLM.ipynb)



contact : bertrand.michel@ec-nantes.fr
