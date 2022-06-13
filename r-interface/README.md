Aymeric Stamm

-   <a href="#setup-instructions-for-using-gudhi-from-r"
    id="toc-setup-instructions-for-using-gudhi-from-r">Setup instructions
    for using GUDHI from R</a>
    -   <a href="#the-reticulate-package" id="toc-the-reticulate-package">The
        <span><strong>reticulate</strong></span> package</a>
    -   <a href="#bring-gudhi-into-r" id="toc-bring-gudhi-into-r">Bring GUDHI
        into R</a>

<!-- README.md is generated from README.qmd. Please edit that file -->

# Setup instructions for using GUDHI from R

<!-- badges: start -->
<!-- badges: end -->

## The [**reticulate**](https://rstudio.github.io/reticulate/) package

The [**reticulate**](https://rstudio.github.io/reticulate/) package
provides a comprehensive set of tools for interoperability between
Python and R. The package includes facilities for:

-   Calling Python from R in a variety of ways including R Markdown,
    sourcing Python scripts, importing Python modules, and using Python
    interactively within an R session.
-   Translation between R and Python objects (for example, between R and
    Pandas data frames, or between R matrices and NumPy arrays).
-   Flexible binding to different versions of Python including virtual
    environments and Conda environments.

[**reticulate**](https://rstudio.github.io/reticulate/) embeds a Python
session within your R session, enabling seamless, high-performance
interoperability. If you are an R developer that uses Python for some of
your work or a member of data science team that uses both languages,
[**reticulate**](https://rstudio.github.io/reticulate/) can dramatically
streamline your workflow.

## Bring GUDHI into R

-   First, you’ll want to install and load the
    [**reticulate**](https://rstudio.github.io/reticulate/) package into
    your session:

    ``` r
    # install.packages("reticulate")
    library(reticulate)
    ```

-   Next, the fastest way to get you set up to use Python from R with
    [**reticulate**](https://rstudio.github.io/reticulate/) is to use
    the `install_miniconda()` utility function which is included in the
    [**reticulate**](https://rstudio.github.io/reticulate/) package. In
    details, use something like:

    ``` r
    install_miniconda()
    ```

-   Next, it is recommended that you create a virtual conda environment
    in which you’ll install all required Python packages. This can be
    achieved using the `reticulate::conda_create()` function as follows:

    ``` r
    version <- "3.9.6"
    conda_create("r-reticulate", python_version = version)
    ```

-   Next, you can seamlessly install all the Python packages you need by
    calling the `conda_install()` function. For instance, to use Gudhi,
    you would do something like:

    ``` r
    conda_install("scikit-learn", envname = "r-reticulate")
    conda_install("gudhi", envname = "r-reticulate")
    ```

This is a setup that you ought to do only once (unless you want to
change your Python version for some reason).

If you want to check your Python configuration, you can do:

``` r
py_config()
#> python:         /Users/stamm-a/Library/r-miniconda/envs/r-reticulate/bin/python3.9
#> libpython:      /Users/stamm-a/Library/r-miniconda/envs/r-reticulate/lib/libpython3.9.dylib
#> pythonhome:     /Users/stamm-a/Library/r-miniconda/envs/r-reticulate:/Users/stamm-a/Library/r-miniconda/envs/r-reticulate
#> version:        3.9.6 | packaged by conda-forge | (default, Jul 11 2021, 03:36:15)  [Clang 11.1.0 ]
#> numpy:          /Users/stamm-a/Library/r-miniconda/envs/r-reticulate/lib/python3.9/site-packages/numpy
#> numpy_version:  1.22.4
#> 
#> NOTE: Python version was forced by RETICULATE_PYTHON
```
