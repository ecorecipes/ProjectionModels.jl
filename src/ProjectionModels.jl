module ProjectionModels

using LinearAlgebra
using Statistics

# Types and traits
include("types.jl")
export AbstractProjectionStructure
export AbstractDensityDependence, DensityIndependent, DensityDependent
export AbstractStochasticity, Deterministic, StochasticKernelResampled, StochasticParameterResampled
export DirectIteration, EigenAnalysis

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

# Matrix properties
include("properties.jl")
export is_irreducible, is_primitive, is_ergodic

# Utilities
include("utils.jl")
export area_under_curve

# Time-lag support
include("time_lag.jl")
export TimeLagStructure
export expand_lag_matrix, extract_lag_components
export augment_population, extract_population
export net_repro_rate_lagged, generation_time_lagged

end # module ProjectionModels
