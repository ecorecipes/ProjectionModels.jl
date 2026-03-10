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

abstract type AbstractDensityDependence end
struct DensityIndependent <: AbstractDensityDependence end
struct DensityDependent <: AbstractDensityDependence end

# --- Stochasticity traits ---

abstract type AbstractStochasticity end
struct Deterministic <: AbstractStochasticity end
struct StochasticKernelResampled <: AbstractStochasticity end
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
