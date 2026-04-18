# Types

Abstract and concrete types for classifying structured population projection models.

## Top-level structure

```@docs
AbstractProjectionStructure
AbstractContinuousStateStructure
AbstractIPMStructure
SimpleIPM
GeneralIPM
SimpleContinuousState
GeneralContinuousState
```

## Density dependence and stochasticity

```@docs
AbstractDensityDependence
DensityIndependent
DensityDependent
AbstractStochasticity
Deterministic
StochasticKernelResampled
StochasticParameterResampled
```

## State and time semantics

```@docs
AbstractStateSemantics
FiniteState
ContinuousState
AbstractTimeSemantics
DiscreteTime
ContinuousTime
```

## Solver / analysis dispatch tags

```@docs
DirectIteration
EigenAnalysis
AbstractProjectionSolution
```

## Delay-generator term

```@docs
DelayGeneratorTerm
```
