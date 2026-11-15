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

## Coupled population systems

The data layer for systems of several named components with auxiliary state, scheduled
events, interaction rules and observables. MatrixProjectionModels.jl and
PhysiologicallyBasedDemographicModels.jl both build on it: they import these names, export
them, and add the solver that steps a system. None of the names is exported from the core
package.

A component carries its numbers in `component.population`. The core handles stage vectors.
A package with another kind of carrier adds methods of `total_population`, `inject!`,
`remove_fraction!` and `as_component` for it.

### Components and systems

```@docs
StructuredPopulationCore.PopulationComponent
StructuredPopulationCore.PopulationSystem
StructuredPopulationCore.AbstractStateVariable
StructuredPopulationCore.get_state
StructuredPopulationCore.has_state
StructuredPopulationCore.total_population
StructuredPopulationCore.component_total
StructuredPopulationCore.component_totals
StructuredPopulationCore.by_species
StructuredPopulationCore.by_type
StructuredPopulationCore.by_patch
StructuredPopulationCore.inject!
StructuredPopulationCore.remove_fraction!
```

### Events, rules and observables

```@docs
StructuredPopulationCore.AbstractScheduledEvent
StructuredPopulationCore.apply_event!
StructuredPopulationCore.PulseRelease
StructuredPopulationCore.SingleDayRelease
StructuredPopulationCore.SprayEvent
StructuredPopulationCore.AbstractInteractionRule
StructuredPopulationCore.TransferRule
StructuredPopulationCore.ReproductionRule
StructuredPopulationCore.MortalityRule
StructuredPopulationCore.CustomRule
StructuredPopulationCore.Observable
```

### Functions the model packages extend

```@docs
StructuredPopulationCore.n_stages
StructuredPopulationCore.allocate
StructuredPopulationCore.supply_demand_index
StructuredPopulationCore.as_component
StructuredPopulationCore.component_stage_names
StructuredPopulationCore.validate_component_model
StructuredPopulationCore.convert_component
```
