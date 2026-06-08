module StructuredPopulationCore

using LinearAlgebra
using Statistics

# Types and traits
include("types.jl")
export AbstractProjectionStructure
export AbstractContinuousStateStructure, AbstractIPMStructure
export SimpleIPM, GeneralIPM
export SimpleContinuousState, GeneralContinuousState
export AbstractTimeSemantics, DiscreteTime, ContinuousTime
export AbstractStateSemantics, FiniteState, ContinuousState
export AbstractDensityDependence, DensityIndependent, DensityDependent
export AbstractStochasticity, Deterministic, StochasticKernelResampled, StochasticParameterResampled
export DirectIteration, EigenAnalysis
export DelayGeneratorTerm

# Shared domains
include("domains.jl")
export AbstractStateDomain, ContinuousDomain, DiscreteDomain
export meshpoints, step_size, bounds, n_states

# Solution interface
include("solution.jl")
export AbstractProjectionSolution

# Eigenanalysis
include("eigenanalysis.jl")
export eigenanalysis_power, eigenanalysis_full

# Analysis functions
include("analysis.jl")
export lambda, stable_distribution, reproductive_value
export sensitivity, elasticity, damping_ratio
export stochastic_growth_rate, mean_kernel

# LTRE analysis
include("ltre.jl")
export LTREResult, StochasticLTREResult, SNALTREResult
export ExactLTREResult, RandomLTREResult
export ltre, stochastic_ltre, sna_ltre
export ltre_random, exact_ltre, gmatrix

# Matrix properties
include("properties.jl")
export is_irreducible, is_primitive, is_ergodic

# Utilities
include("utils.jl")
export area_under_curve

# State block layouts
include("state_blocks.jl")
export StateBlockLayout
export blocknames, blockrange, blockranges
export split_state, combine_state

# Time-lag support
include("time_lag.jl")
export TimeLagStructure
export expand_lag_matrix, extract_lag_components
export augment_population, extract_population
export net_repro_rate_lagged, generation_time_lagged

end # module StructuredPopulationCore
