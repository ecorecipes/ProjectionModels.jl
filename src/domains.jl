"""
Shared state-domain types for projection and population-dynamics packages.
"""

"""
    AbstractStateDomain

Supertype for shared domain descriptors over which state variables are defined.
"""
abstract type AbstractStateDomain end

"""
    ContinuousDomain{T<:Real}

Represents a continuous state variable domain discretized via the midpoint
rule.

The domain `[lower, upper]` is divided into `n_meshpoints` bins. Meshpoints are
the midpoints of each bin.
"""
struct ContinuousDomain{T<:Real} <: AbstractStateDomain
    lower::T
    upper::T
    n_meshpoints::Int

    function ContinuousDomain(lower::T, upper::T, n_meshpoints::Int) where {T<:Real}
        lower < upper || throw(ArgumentError("lower must be less than upper"))
        n_meshpoints > 0 || throw(ArgumentError("n_meshpoints must be positive"))
        new{T}(lower, upper, n_meshpoints)
    end
end

function ContinuousDomain(lower::Real, upper::Real, n_meshpoints::Int)
    T = promote_type(typeof(lower), typeof(upper))
    ContinuousDomain(T(lower), T(upper), n_meshpoints)
end

"""
    step_size(d::ContinuousDomain)

Width of each discretization bin: `(upper - lower) / n_meshpoints`.
"""
step_size(d::ContinuousDomain) = (d.upper - d.lower) / d.n_meshpoints

"""
    meshpoints(d::ContinuousDomain)

Return the midpoints of the `n_meshpoints` bins spanning `[lower, upper]`.
"""
function meshpoints(d::ContinuousDomain{T}) where {T}
    n = d.n_meshpoints
    domain_bounds = range(d.lower, d.upper; length = n + 1)
    return [(domain_bounds[i] + domain_bounds[i + 1]) / 2 for i in 1:n]
end

"""
    bounds(d::ContinuousDomain)

Return the `n_meshpoints + 1` bin edges spanning `[lower, upper]`.
"""
function bounds(d::ContinuousDomain)
    collect(range(d.lower, d.upper; length = d.n_meshpoints + 1))
end

"""
    DiscreteDomain

Represents a discrete state variable with named levels.
"""
struct DiscreteDomain <: AbstractStateDomain
    labels::Vector{Symbol}
end

n_states(d::DiscreteDomain) = length(d.labels)
n_states(d::ContinuousDomain) = d.n_meshpoints
