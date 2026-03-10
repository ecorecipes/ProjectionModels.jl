# ProjectionModels.jl

Shared abstractions for structured population projection models in Julia. This package provides the common types, eigenanalysis routines, and analysis functions used by [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl), [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl), and [CategoricalProjectionModels.jl](https://github.com/ecorecipes/CategoricalProjectionModels.jl).

## Features

- **Eigenanalysis**: `lambda`, `stable_distribution`, `reproductive_value`, `sensitivity`, `elasticity`, `damping_ratio`
- **Matrix properties**: `is_irreducible`, `is_primitive`, `is_ergodic`
- **Time-lag support**: `TimeLagStructure`, `expand_lag_matrix`, `extract_lag_components`
- **Stochastic growth**: `stochastic_growth_rate`, `mean_kernel`
- **Shared type hierarchy**: `DensityIndependent`/`DensityDependent`, `Deterministic`/`Stochastic*`, `DirectIteration`/`EigenAnalysis`

## Installation

```julia
using Pkg
Pkg.add("ProjectionModels")
```

## Related

- [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl) — discrete-stage matrix models
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl) — continuous-state integral projection models
- [CategoricalProjectionModels.jl](https://github.com/ecorecipes/CategoricalProjectionModels.jl) — categorical/functorial framework
