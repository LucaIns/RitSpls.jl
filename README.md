# `RitSpls`: A Julia package for Robustness-Inducing Transformations in SPLS regression


`RitSpls` implements a robustification of Sparse Partial Least Squares (SPLS) through robustness-inducing transformations on the (univariate) predictand and predictors. 

The approach considered in \[1\] is:

1. transform multivariate predictors through a Generalized Spatial Sign Pre-Processing (GSSPP) \[2\];

2. transform the univariate predictand through a Wrapping transformation \[3\];

3. use the SNIPLS (sparse NIPALS) algorithm \[4\] to estimate an SPLS model from the (robustly) transformed inputs.

The package provides an `SPLS` class that allows to interface with the ScikitLearn API. 
For instance, objects of the `SPLS` and `GSSPP` class accept widely used ScikitLearn functions and routines (e.g, `fit!`, `predict`, `transform`, `GridSearchCV`, etc.).


How to install
--------------
`]add <path to this GitHub repo>`

Examples
--------
Examples on how to use `RitSpls`are presented as Jupyter notebooks in the [documentation](https://github.com/LucaIns/RitSpls.jl/tree/main/doc) folder.


References
---------

\[1\] Serneels, S., L. Insolia, and T. Verdonck (2023). “Elegant robustfication of sparse partial least squares by robustness-inducing transformations". Submitted.

\[2\] Raymaekers, J. and P. J. Rousseeuw (2019). “A generalized spatial sign covariance matrix”. In: Journal of Multivariate Analysis 171, pp. 94–111.

\[3\] Raymaekers, J. and P. J. Rousseeuw (2021). “Fast robust correlation for high-dimensional data”. In: Technometrics 63 (2), pp. 184–198.

\[4\] Hoffmann, I., P. Filzmoser, S. Serneels, and K. Varmuza (2016). “Sparse and robust PLS for binary classification”. In: Journal of Chemometrics 30.4, pp. 153–162.


Credits
--------
Functions for robustness-inducing transformations are adapted from R code developed by ‪Jakob Raymaekers‬.
