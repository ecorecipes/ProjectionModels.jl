# Shared Operations and Verbs

Functions that several packages of the stack implement or extend. They are defined once,
here, so that every package adds methods to the same function instead of defining another
function of the same name. The names can then be used unqualified whichever packages are
loaded together.

## Kernels, matrices and block matrices

Methods for the core types. IntegralProjectionModels.jl re-exports them, and
CategoricalPopulationDynamics.jl extends them to its own domains, to projection nets and
to Catlab's `FinFunction`.

```@docs
left_kan_extension
StructuredPopulationCore.discretise_kernel
right_kan_extension
stratify
coarsen
compose_from_uwd
```

## Verbs implemented by the model packages

Declared here without methods. Each model package adds the method for its own problem
type. `remake` is shared in the same way, through `SciMLBase.remake`.

```@docs
demographic_ensemble
to_discrete_problem
to_ode_problem
to_dde_problem
StructuredPopulationCore.set_state!
```
