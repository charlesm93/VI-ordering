# Ordering of divergences

This repo contains code to reproduce the numerical experiments and figures in the paper "An Ordering of divergences for variational inference with a factorized Gaussian approximation" by Charles Margossian, Loucas Pillaid-Vivien and Lawrence Saul.

The files are organized as follows:
* `vi_ordering.r`, with support from `fgvi_solve.r`, contains the code to reproduce Figure 1 (ellipse and circle plots) and Figure 3 (variance, precision and entropy estimate as we violate the assumption of factorization).
* `entropy_matching.r` contains the code to reproduce Figure 4 (alpha for entropy match).
* `ordering.ipynb`, with support from `fgvi.py` and `factirized_bam.py`, contains the code to reproduce Figure 5 (variance estimates for the 8 schools) and Figure 6 (entropies for inference gym model). This notebook contains additional code to examine various properties of the VI algorithms, including convergence.


