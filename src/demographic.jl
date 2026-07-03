"""
Demographic stochasticity primitives.

Two layers are provided:

1. A discrete-time, integer-count update [`demographic_step!`](@ref) that realizes
   one step of a structured projection as a Multinomial survival/movement draw plus
   a Poisson fecundity draw, with `E[n_{t+1} | n_t] = (transition + fecundity) * n_t`
   — i.e. the deterministic operator. This is the primitive used by the discrete-time
   backends (matrix and binned-IPM).

2. A general reaction intermediate representation (IR) — [`DemographicReaction`](@ref)
   and [`DemographicReactionSystem`](@ref) — together with a reference continuous-time
   realizer [`gillespie`](@ref). This is the substrate the continuous-time backends
   (finite-state CTMC, PSPM/PDMP) build on.

The samplers here are dependency-light (Knuth's Poisson for small means, a Normal
approximation for large means). For exact large-mean sampling, a consumer holding
`Distributions.jl` can supply its own draws.
"""

# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------

"""
    rand_poisson(rng, λ)

Draw a `Poisson(λ)` integer. Uses Knuth's exact algorithm for `λ < 30` and a
Normal approximation (matching mean and variance, rounded to a nonnegative
integer) for larger `λ`.
"""
function rand_poisson(rng::AbstractRNG, λ::Real)
    λ <= 0 && return 0
    if λ < 30
        L = exp(-λ)
        k = 0
        p = 1.0
        while true
            k += 1
            p *= rand(rng)
            p <= L && return k - 1
        end
    else
        return max(0, round(Int, λ + sqrt(λ) * randn(rng)))
    end
end

# ----------------------------------------------------------------------------
# Discrete-time integer-count update
# ----------------------------------------------------------------------------

"""
    rand_binomial(rng, n, p)

Draw a `Binomial(n, p)` integer. Exact (sum of Bernoullis) for `n ≤ 64`, and a
Normal approximation (matching mean and variance, rounded and clamped to
`[0, n]`) for larger `n` — so a super-individual cohort of `n` shares one draw.
"""
function rand_binomial(rng::AbstractRNG, n::Integer, p::Real)
    (n <= 0 || p <= 0) && return 0
    p >= 1 && return Int(n)
    if n <= 64
        c = 0
        for _ in 1:n
            rand(rng) < p && (c += 1)
        end
        return c
    else
        μ = n * p
        σ = sqrt(n * p * (1 - p))
        return clamp(round(Int, μ + σ * randn(rng)), 0, Int(n))
    end
end

"""
    demographic_step!(rng, n_next, n, transition, fecundity)

One demographic-stochastic update of the integer count vector `n` into `n_next`.

- `transition` is sub-stochastic (each column sums to ≤ 1): the `n[j]` individuals
  in source class `j` are partitioned by a `Multinomial` draw across destinations
  `transition[:, j]`, with the deficit `1 - Σ transition[:, j]` accounting for
  mortality.
- `fecundity[i, j]` is the expected number of class-`i` offspring per class-`j`
  individual; offspring are added as `Poisson(fecundity[i, j] * n[j])`.

The conditional mean is `E[n_next | n] = (transition + fecundity) * n`, so the
update is an unbiased finite-population realization of the deterministic operator.

The `Multinomial` partition is realized via sequential conditional `Binomial`
draws (the standard "stick-breaking" construction: destination `i` draws
`Binomial(remaining, transition[i,j] / (1 - Σ_{i'<i} transition[i',j]))`
individuals from those not yet assigned) rather than one categorical draw per
individual, so a source class of `n[j]` individuals costs `O(k)` `Binomial`
draws instead of `O(n[j])` — identical distribution, no per-individual loop.

Returns `n_next`.
"""
function demographic_step!(rng::AbstractRNG, n_next::AbstractVector{<:Integer},
                           n::AbstractVector{<:Integer},
                           transition::AbstractMatrix, fecundity::AbstractMatrix)
    k = length(n)
    fill!(n_next, 0)
    # Survival / movement: sequential-Binomial ("stick-breaking") realization of
    # the Multinomial partition of each source class.
    @inbounds for j in 1:k
        remaining = Int(n[j])
        remaining == 0 && continue
        cum = 0.0
        for i in 1:k
            remaining == 0 && break
            pij = transition[i, j]
            if pij > 0
                denom = 1.0 - cum
                q = denom > 0 ? clamp(pij / denom, 0.0, 1.0) : 1.0
                drawn = rand_binomial(rng, remaining, q)
                if drawn > 0
                    n_next[i] += drawn
                    remaining -= drawn
                end
            end
            cum += pij
        end
        # any individuals left in `remaining` after all k destinations = mortality
    end
    # Fecundity: Poisson offspring.
    @inbounds for j in 1:k
        nj = n[j]
        nj == 0 && continue
        for i in 1:k
            fij = fecundity[i, j]
            fij > 0 && (n_next[i] += rand_poisson(rng, fij * nj))
        end
    end
    return n_next
end

"""
    demographic_step(rng, n, transition, fecundity)

Allocating form of [`demographic_step!`](@ref).
"""
function demographic_step(rng::AbstractRNG, n::AbstractVector,
                          transition::AbstractMatrix, fecundity::AbstractMatrix)
    n_next = zeros(Int, length(n))
    demographic_step!(rng, n_next, collect(Int, n), transition, fecundity)
    return n_next
end

# ----------------------------------------------------------------------------
# Reaction intermediate representation (IR)
# ----------------------------------------------------------------------------

"""
    DemographicReaction(propensity, stoichiometry)
    DemographicReaction(propensity, n_states, (i => δ)...)

A single demographic event. `propensity` is a nonnegative rate, either a number
or a callable `(n, p, t) -> Real`. `stoichiometry::Vector{Int}` is the change
applied to the integer count vector when the reaction fires; the sparse form sets
`stoichiometry[i] += δ` for each `i => δ`.
"""
struct DemographicReaction{P}
    propensity::P
    stoichiometry::Vector{Int}
end

function DemographicReaction(propensity, n_states::Integer, changes::Pair{<:Integer,<:Integer}...)
    s = zeros(Int, n_states)
    for (i, δ) in changes
        s[i] += δ
    end
    return DemographicReaction(propensity, s)
end

"""
    DemographicReactionSystem(n_states, reactions)

A collection of [`DemographicReaction`](@ref)s over an `n_states`-dimensional
integer count vector. This is the shared IR consumed by demographic realizers —
the continuous-time [`gillespie`](@ref) here, and discrete-time / PDMP realizers
in the backend packages.
"""
struct DemographicReactionSystem{R<:DemographicReaction}
    n_states::Int
    reactions::Vector{R}
end

@inline _propensity(a::Number, n, p, t) = a
@inline _propensity(a, n, p, t) = a(n, p, t)

"""
    propensities!(out, sys, n, p, t)

Fill `out` with the propensity of each reaction in `sys` at state `n`.
"""
function propensities!(out::AbstractVector, sys::DemographicReactionSystem, n, p, t)
    @inbounds for i in eachindex(sys.reactions)
        out[i] = _propensity(sys.reactions[i].propensity, n, p, t)
    end
    return out
end

"""
    total_propensity(sys, n, p, t)

Sum of all reaction propensities at state `n`.
"""
total_propensity(sys::DemographicReactionSystem, n, p, t) =
    sum(_propensity(rx.propensity, n, p, t) for rx in sys.reactions; init = 0.0)

"""
    apply_reaction!(n, reaction)

Apply a reaction's stoichiometry to the count vector `n` in place.
"""
function apply_reaction!(n::AbstractVector{<:Integer}, rx::DemographicReaction)
    n .+= rx.stoichiometry
    return n
end

"""
    gillespie(rng, sys, n0, tspan; p=nothing, max_events=1_000_000)

Reference continuous-time realizer (Gillespie's direct method) for a
[`DemographicReactionSystem`](@ref). Returns `(ts, us)` — the event times and the
integer count vectors after each event (including the initial state). Stops at
`tspan[2]`, when all propensities vanish (an absorbing state), or after
`max_events`.

Propensities are assumed time-homogeneous between events; explicitly
time-dependent rates require thinning, which this reference realizer does not do.
"""
function gillespie(rng::AbstractRNG, sys::DemographicReactionSystem, n0, tspan;
                   p = nothing, max_events::Int = 1_000_000)
    t = float(tspan[1])
    tend = float(tspan[2])
    n = collect(Int, n0)
    ts = [t]
    us = [copy(n)]
    a = zeros(Float64, length(sys.reactions))
    for _ in 1:max_events
        propensities!(a, sys, n, p, t)
        a0 = sum(a)
        a0 <= 0 && break
        t += -log(rand(rng)) / a0
        t > tend && break
        threshold = rand(rng) * a0
        cum = 0.0
        j = length(a)
        for i in eachindex(a)
            cum += a[i]
            if threshold < cum
                j = i
                break
            end
        end
        apply_reaction!(n, sys.reactions[j])
        push!(ts, t)
        push!(us, copy(n))
    end
    return ts, us
end

# ----------------------------------------------------------------------------
# Generator -> reactions
# ----------------------------------------------------------------------------

# First-order reaction propensity with rate coefficient `coef` on state `i`.
# A fresh closure per call avoids loop-capture issues.
_linear_propensity(coef, i) = (n, p, t) -> coef * n[i]

"""
    generator_reactions(G; source=nothing)

Build a `DemographicReactionSystem` from an `n × n` generator `G` whose
conditional mean reproduces `dn/dt = G·n` exactly. Off-diagonal entries
`G[j,i] > 0` become **migration** reactions `i → j` (`-eᵢ + eⱼ`); each state's
net column sum becomes a self **birth** (`+eᵢ`, if positive) or **death**
(`-eᵢ`, if negative); a constant nonnegative `source` vector adds immigration.

The mean is exact for any generator. The *fluctuation* structure follows this
migration-plus-net-birth/death convention — the correct continuous-time Markov
chain when off-diagonal flows are genuine movements. For models where
off-diagonal entries are fecundity (a parent persists while producing
offspring), build the reactions explicitly so the stoichiometry is right.
"""
function generator_reactions(G::AbstractMatrix; source = nothing)
    n = size(G, 1)
    size(G, 2) == n || throw(DimensionMismatch("generator must be square; got $(size(G))"))
    reactions = DemographicReaction[]
    for i in 1:n, j in 1:n
        i == j && continue
        r = G[j, i]
        if r > 0
            push!(reactions, DemographicReaction(_linear_propensity(r, i), n, i => -1, j => +1))
        elseif r < 0
            throw(ArgumentError(
                "off-diagonal generator entry G[$j,$i] = $r is negative; not a valid rate"))
        end
    end
    for i in 1:n
        cs = sum(@view G[:, i])
        if cs > 0
            push!(reactions, DemographicReaction(_linear_propensity(cs, i), n, i => +1))
        elseif cs < 0
            push!(reactions, DemographicReaction(_linear_propensity(-cs, i), n, i => -1))
        end
    end
    if source !== nothing
        src = collect(source)
        length(src) == n || throw(DimensionMismatch(
            "source length $(length(src)) does not match generator size $n"))
        for i in 1:n
            si = src[i]
            if si > 0
                push!(reactions, DemographicReaction(float(si), n, i => +1))   # immigration
            elseif si < 0
                throw(ArgumentError("source[$i] = $si < 0 is not a valid immigration rate"))
            end
        end
    end
    return DemographicReactionSystem(n, reactions)
end

# ----------------------------------------------------------------------------
# Chemical Langevin equation (diffusion approximation of demographic noise)
# ----------------------------------------------------------------------------

"""
    num_reactions(sys)

Number of reactions (independent demographic noise channels) in `sys`.
"""
num_reactions(sys::DemographicReactionSystem) = length(sys.reactions)

"""
    cle_drift!(du, sys, n, p, t)

Chemical Langevin drift `Σᵣ νᵣ aᵣ(n)` — equal to the deterministic macroscopic
RHS of the reaction system.
"""
function cle_drift!(du, sys::DemographicReactionSystem, n, p, t)
    fill!(du, zero(eltype(du)))
    for rx in sys.reactions
        a = _propensity(rx.propensity, n, p, t)
        @. du += a * rx.stoichiometry
    end
    return du
end

"""
    cle_noise!(M, sys, n, p, t)

Fill the `(n_states × num_reactions)` chemical Langevin diffusion matrix
`M[i, r] = νᵣ[i] · √(max(aᵣ(n), 0))`. The CLE is
`dn = cle_drift dt + M dW`, with one independent Brownian motion per reaction;
`M Mᵀ = ν diag(a) νᵀ` is the demographic diffusion (variance ∝ system size).
"""
function cle_noise!(M, sys::DemographicReactionSystem, n, p, t)
    fill!(M, zero(eltype(M)))
    for (r, rx) in enumerate(sys.reactions)
        a = _propensity(rx.propensity, n, p, t)
        s = sqrt(max(a, zero(a)))
        @. M[:, r] = rx.stoichiometry * s
    end
    return M
end
