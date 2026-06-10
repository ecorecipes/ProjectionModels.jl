"""
Shared abstract types and trait types for projection models.
"""

# --- Abstract supertypes for package-specific structure types ---

"""
    AbstractProjectionStructure

Supertype for projection model structure traits.
Subtypes include shared continuous-state structures plus
package-specific specializations.
"""
abstract type AbstractProjectionStructure end

"""
    AbstractContinuousStateStructure

Shared supertype for continuous-state model structures. This is used by both
discrete-time integral projection models and the forthcoming continuous-state
continuous-time backends.
"""
abstract type AbstractContinuousStateStructure <: AbstractProjectionStructure end

"""
    AbstractIPMStructure

Compatibility supertype for the existing integral projection model structure
names. It remains available as part of the shared abstraction layer so current
IPM code can keep its API while new packages dispatch on the more neutral
`AbstractContinuousStateStructure`.
"""
abstract type AbstractIPMStructure <: AbstractContinuousStateStructure end

"""
    SimpleIPM

Single continuous-state structure.
"""
struct SimpleIPM <: AbstractIPMStructure end

"""
    GeneralIPM

Multi-state or otherwise generalized continuous-state structure.
"""
struct GeneralIPM <: AbstractIPMStructure end

"""
    SimpleContinuousState

Alias of [`SimpleIPM`](@ref). Use this name when writing code at the
generic continuous-state-structure level (e.g., for finite-element or
non-IPM continuous-state schemes).
"""
const SimpleContinuousState = SimpleIPM

"""
    GeneralContinuousState

Alias of [`GeneralIPM`](@ref). Use this name when writing code at the
generic multi-state continuous-state level.
"""
const GeneralContinuousState = GeneralIPM

# --- Time semantics traits ---

"""
    AbstractTimeSemantics

Supertype for time-semantics traits.
"""
abstract type AbstractTimeSemantics end

"""
    DiscreteTime

One-step update semantics over a discrete clock.
"""
struct DiscreteTime <: AbstractTimeSemantics end

"""
    ContinuousTime

Infinitesimal-generator semantics over continuous time.
"""
struct ContinuousTime <: AbstractTimeSemantics end

# --- State space semantics traits ---

"""
    AbstractStateSemantics

Supertype for state-space semantics traits.
"""
abstract type AbstractStateSemantics end

"""
    FiniteState

Finite-dimensional or explicitly enumerated state space.
"""
struct FiniteState <: AbstractStateSemantics end

"""
    ContinuousState

Continuum-valued or discretized-from-continuum state space.
"""
struct ContinuousState <: AbstractStateSemantics end

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

"""
    Demographic

Demographic stochasticity: finite-population randomness in which individual fates
(survival, movement, reproduction) are realized as integer-valued random draws
whose conditional mean reproduces the deterministic operator. Realized per time
step for discrete-time models (multinomial survival + Poisson fecundity) and as a
continuous-time Markov jump process for continuous-time models. See
[`demographic_step!`](@ref) and [`DemographicReactionSystem`](@ref).
"""
struct Demographic <: AbstractStochasticity end

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

"""
    DelayGeneratorTerm(lag, operator)

A shared delayed linear contribution for continuous-time generator formulations.
The operator may be a matrix or a callable returning one.
"""
struct DelayGeneratorTerm{T<:Real, O}
    lag::T
    operator::O

    function DelayGeneratorTerm{T, O}(lag::T, operator::O) where {T<:Real, O}
        lag > zero(T) || throw(ArgumentError("lag must be positive"))
        new{T, O}(lag, operator)
    end
end

function DelayGeneratorTerm(lag::Real, operator)
    T = typeof(float(lag))
    DelayGeneratorTerm{T, typeof(operator)}(T(lag), operator)
end
