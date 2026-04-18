# State Domains and Block Layouts

Helpers for describing the discretised state space of a model and for
splitting / combining flat state vectors into named per-state-variable
blocks. These types are used throughout the `*ProjectionModels` and
`*PopulationDynamics` packages to bridge between the abstract model
description and the numeric state representation passed to solvers.

## State domains

A state domain describes how the trait or stage axis is discretised.

```@docs
AbstractStateDomain
ContinuousDomain
DiscreteDomain
meshpoints
bounds
step_size
n_states
```

## Block layouts for multi-state systems

`StateBlockLayout` partitions a flat state vector into named contiguous
blocks, one per state variable. The companion functions return the
indices (`blockrange`, `blockranges`), names (`blocknames`), and split /
combine the underlying numeric vector.

```@docs
StateBlockLayout
blocknames
blockrange
blockranges
split_state
combine_state
```
