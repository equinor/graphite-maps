# graphite-maps

Graph informed triangular ensemble-to-posterior maps

## Installation

This library depends on
[scikit-sparse](https://github.com/scikit-sparse/scikit-sparse),
which depends on `suitesparse`.
Make sure that the system `suitesparse` matches `scikit-sparse`.
See Github Actions CI files for more information.

## Earlier version

If you wish to study the code as it was around the time the 
[accompanying paper](https://arxiv.org/abs/2501.09016)
was put on ArXiv, have a look at commit
[cb839...](https://github.com/equinor/graphite-maps/tree/cb83947365c3e46ea9e7d885ebeda45fb4591791).
Since then, bug have been fixed and improvements have been made, but if you are reading the paper
and want to e.g. reproduce the figures, then going back to that commit might be useful.

## References

- [An Ensemble Information Filter: Retrieving Markov-information from the SPDE discretisation](https://arxiv.org/abs/2501.09016)
