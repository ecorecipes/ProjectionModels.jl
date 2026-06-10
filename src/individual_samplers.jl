"""
Individual-level vital-rate samplers for individual-based (agent / ECS)
realizations of population models.

These turn deterministic vital-rate descriptions — survival probabilities,
growth and recruit distributions, fecundity rates — into random draws for a
single individual of trait `z`. Backends provide analytic methods for their
parametric vital-rate types; the generic fallbacks here work for any callable
density given a `ContinuousDomain` (discretized inverse-CDF), which is also the
bridge to the binned demographic representation.

Conventions:
- `survival` is a probability callable `s(z)`.
- `growth`/`fecundity` are kernel densities `k(z_new, z)` (offspring-trait first).
- a recruit's trait is drawn from the offspring-trait marginal of `fecundity`.
"""

"""
    sample_survives(rng, survival, z) -> Bool

Bernoulli survival draw with probability `clamp(survival(z), 0, 1)`.
"""
sample_survives(rng::AbstractRNG, survival, z) = rand(rng) < clamp(survival(z), 0, 1)

"""
    sample_from_density(rng, density, domain::ContinuousDomain)

Draw a value in `domain` from a (possibly unnormalized) nonnegative `density(x)`
by inverse-CDF sampling on the midpoint mesh, with uniform jitter within the
chosen bin. Generic fallback when no analytic sampler is available.
"""
function sample_from_density(rng::AbstractRNG, density, domain::ContinuousDomain)
    zs = meshpoints(domain)
    h = step_size(domain)
    m = length(zs)
    total = 0.0
    @inbounds for i in 1:m
        total += max(density(zs[i]), 0.0)
    end
    total > 0 || return zs[rand(rng, 1:m)]
    target = rand(rng) * total
    cum = 0.0
    @inbounds for i in 1:m
        cum += max(density(zs[i]), 0.0)
        if target <= cum
            return zs[i] + (rand(rng) - 0.5) * h
        end
    end
    return zs[m]
end

"""
    sample_growth(rng, growth, z, domain) -> trait

Sample the next trait of a survivor growing from `z` (a draw from `g(·|z)`).
Generic fallback samples the density `x -> growth(x, z)` on `domain`; backends
override for parametric growth types.
"""
sample_growth(rng::AbstractRNG, growth, z, domain::ContinuousDomain) =
    sample_from_density(rng, x -> growth(x, z), domain)

"""
    expected_offspring(fecundity, z, domain) -> Real

Expected number of offspring from an individual of trait `z`. Generic fallback
integrates `∫ fecundity(z_new, z) dz_new` over `domain` (midpoint); backends
override for parametric fecundity types.
"""
function expected_offspring(fecundity, z, domain::ContinuousDomain)
    zs = meshpoints(domain)
    s = 0.0
    @inbounds for zp in zs
        s += fecundity(zp, z)
    end
    return step_size(domain) * s
end

"""
    offspring_count(rng, fecundity, z, domain) -> Int

Number of offspring from an individual of trait `z`: `Poisson(expected_offspring)`.
"""
offspring_count(rng::AbstractRNG, fecundity, z, domain::ContinuousDomain) =
    rand_poisson(rng, expected_offspring(fecundity, z, domain))

"""
    sample_recruit(rng, fecundity, z, domain) -> trait

Sample the trait of a new recruit from a parent of trait `z`. Generic fallback
samples from `x -> fecundity(x, z)` (the offspring-trait marginal) on `domain`;
backends override for parametric fecundity types.
"""
sample_recruit(rng::AbstractRNG, fecundity, z, domain::ContinuousDomain) =
    sample_from_density(rng, x -> fecundity(x, z), domain)
