"""
Shared abstract types and trait types for projection models.
"""

# --- Abstract supertypes for package-specific structure types ---

"""
    AbstractProjectionStructure

Supertype for projection model structure traits.
Subtypes: `AbstractIPMStructure` (in IntegralProjectionModels),
          `AbstractMPMStructure` (in MatrixProjectionModels).
"""
abstract type AbstractProjectionStructure end

# --- Density dependence traits ---

"""
    AbstractDensityDependence

Supertype for density dependence traits. Subtypes: [`DensityIndependent`](@ref), [`DensityDependent`](@ref).
"""
abstract type AbstractDensityDependence end

"""
    DensityIndependent

Vital rates do not depend on population size or density.
"""
struct DensityIndependent <: AbstractDensityDependence end

"""
    DensityDependent

Vital rates depend on current population size or density.
"""
struct DensityDependent <: AbstractDensityDependence end

# --- Stochasticity traits ---

"""
    AbstractStochasticity

Supertype for stochasticity traits. Subtypes: [`Deterministic`](@ref), [`StochasticKernelResampled`](@ref), [`StochasticParameterResampled`](@ref).
"""
abstract type AbstractStochasticity end

"""
    Deterministic

No stochastic variation; vital rates and kernels are fixed across time steps.
"""
struct Deterministic <: AbstractStochasticity end

"""
    StochasticKernelResampled

Environmental stochasticity via resampling entire kernel matrices at each time step.
"""
struct StochasticKernelResampled <: AbstractStochasticity end

"""
    StochasticParameterResampled

Environmental stochasticity via resampling parameters that generate kernels at each time step.
"""
struct StochasticParameterResampled <: AbstractStochasticity end

# --- Algorithm types ---

"""
    DirectIteration

Iterate the population forward via matrix multiplication: `n_{t+1} = K * n_t`.
"""
struct DirectIteration end

"""
    EigenAnalysis

Compute eigendecomposition of the materialized kernel matrix.
No iteration is performed; returns asymptotic quantities directly.
"""
struct EigenAnalysis end
