# ProjectionModels.jl

Shared abstractions for structured population projection models in Julia.

## Overview

ProjectionModels.jl provides the common type hierarchy, eigenanalysis routines, matrix diagnostics, and utility functions used across the [ecorecipes](https://github.com/ecorecipes) family of population modeling packages. It is not typically used on its own but serves as the foundation for higher-level packages.

Key capabilities include:

- **Type hierarchy** for classifying projection models by density dependence, stochasticity, and solution method.
- **Eigenanalysis** via power iteration or full decomposition for asymptotic growth rates, stable distributions, and reproductive values.
- **Sensitivity and elasticity** analysis of projection kernels.
- **Matrix property checks** for irreducibility, primitivity, and ergodicity.
- **Time-lag support** for age-structured or delay models.
- **Stochastic growth rate** estimation for environmentally varying models.

## Quick Example

```julia
using ProjectionModels, LinearAlgebra

# A simple 2-stage projection matrix
A = [0.0  3.0;
     0.5  0.8]

# Asymptotic population growth rate
lam = lambda(A)

# Stable stage distribution and reproductive value
w = stable_distribution(A)
v = reproductive_value(A)

# Sensitivity and elasticity matrices
S = sensitivity(A)
E = elasticity(A)
```

## Related Packages

- [MatrixPopulationModels.jl](https://github.com/ecorecipes/MatrixPopulationModels.jl) -- matrix population models (MPMs)
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl) -- integral projection models (IPMs)
- [CompositePopulationModels.jl](https://github.com/ecorecipes/CompositePopulationModels.jl) -- composite population models (CPMs)
