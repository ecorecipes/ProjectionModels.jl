# StructuredPopulationCore.jl

Shared abstractions for structured population modeling in Julia. This package provides the common types, eigenanalysis routines, and analysis functions used by [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl), [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl), [FiniteStatePopulationDynamics.jl](https://github.com/ecorecipes/FiniteStatePopulationDynamics.jl), [ContinuousStatePopulationDynamics.jl](https://github.com/ecorecipes/ContinuousStatePopulationDynamics.jl), and [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl).

## Features

- **Eigenanalysis**: `lambda`, `stable_distribution`, `reproductive_value`, `sensitivity`, `elasticity`, `damping_ratio`
- **Matrix properties**: `is_irreducible`, `is_primitive`, `is_ergodic`
- **Time-lag support**: `TimeLagStructure`, `expand_lag_matrix`, `extract_lag_components`
- **Stochastic growth**: `stochastic_growth_rate`, `mean_kernel`
- **Shared type hierarchy**: `DensityIndependent`/`DensityDependent`, `Deterministic`/`Stochastic*`, `DirectIteration`/`EigenAnalysis`

## Installation

This package is not yet registered in the Julia General registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/StructuredPopulationCore.jl")
```

## Related

- [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl) — discrete-stage matrix models
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl) — continuous-state integral projection models
- [FiniteStatePopulationDynamics.jl](https://github.com/ecorecipes/FiniteStatePopulationDynamics.jl) — finite-state continuous-time dynamics
- [ContinuousStatePopulationDynamics.jl](https://github.com/ecorecipes/ContinuousStatePopulationDynamics.jl) — continuous-state continuous-time dynamics
- [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl) — categorical/functorial framework
- [PhysiologicallyBasedDemographicModels.jl](https://github.com/ecorecipes/PhysiologicallyBasedDemographicModels.jl) — application-level PBDM reference suite
