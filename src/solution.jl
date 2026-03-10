"""
Abstract solution type and accessor interface for projection model solutions.
"""

"""
    AbstractProjectionSolution

Supertype for projection model solutions. Subtypes must have fields:
- `t`: time steps
- `u`: population states
- `kernel_matrices`: materialized kernel(s)
- `eigenanalysis`: named tuple `(lambda, stable_dist, repro_value)` or `nothing`
- `retcode`: Symbol
- `lambdas`: per-step growth rates
"""
abstract type AbstractProjectionSolution end
