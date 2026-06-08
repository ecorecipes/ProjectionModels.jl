"""
Quasi-extinction analysis for stochastic population projections.

Computes extinction risk metrics from stochastic simulation results.

Reference: Morris & Doak (2002) Quantitative Conservation Biology.
"""

"""
    QuasiExtinctionResult

Result of quasi-extinction analysis.

# Fields
- `threshold`: Population size threshold for quasi-extinction
- `prob_extinct`: Probability of reaching below threshold by end of simulation
- `mean_time_to_extinction`: Mean time to first crossing below threshold (NaN if never)
- `extinction_times`: Vector of extinction times per replicate (Inf if survived)
- `cumulative_risk`: Cumulative probability of extinction at each time step
"""
struct QuasiExtinctionResult{T<:Real}
    threshold::T
    prob_extinct::T
    mean_time_to_extinction::T
    extinction_times::Vector{T}
    cumulative_risk::Vector{T}
end

"""
    quasi_extinction(trajectories::AbstractMatrix; threshold=1.0)

Analyze quasi-extinction risk from population trajectories.

# Arguments
- `trajectories`: Matrix of size (n_time × n_reps) with total population sizes
- `threshold`: Population size below which quasi-extinction is declared

# Returns
A `QuasiExtinctionResult` with extinction probabilities and timing.
"""
function quasi_extinction(trajectories::AbstractMatrix; threshold::Real=1.0)
    n_time, n_reps = size(trajectories)
    T = Float64

    # Find first extinction time for each replicate
    ext_times = fill(Inf, n_reps)
    for rep in 1:n_reps
        for t in 1:n_time
            if trajectories[t, rep] < threshold
                ext_times[rep] = T(t)
                break
            end
        end
    end

    # Probability of extinction by end
    n_extinct = count(isfinite, ext_times)
    prob_extinct = n_extinct / n_reps

    # Mean time to extinction (only for those that went extinct)
    finite_times = filter(isfinite, ext_times)
    mean_ext_time = isempty(finite_times) ? T(NaN) : mean(finite_times)

    # Cumulative risk curve
    cumulative = zeros(T, n_time)
    for t in 1:n_time
        cumulative[t] = count(x -> x <= t, ext_times) / n_reps
    end

    return QuasiExtinctionResult{T}(T(threshold), prob_extinct,
                                    mean_ext_time, ext_times, cumulative)
end

"""
    quasi_extinction(pop_sizes::AbstractVector{<:AbstractVector}; threshold=1.0)

Analyze quasi-extinction from a vector of population size trajectories
(one vector per replicate).
"""
function quasi_extinction(pop_sizes::AbstractVector{<:AbstractVector}; threshold::Real=1.0)
    n_reps = length(pop_sizes)
    n_time = maximum(length, pop_sizes)

    # Pad shorter trajectories with their last value
    trajectories = zeros(Float64, n_time, n_reps)
    for (rep, traj) in enumerate(pop_sizes)
        for t in 1:length(traj)
            trajectories[t, rep] = traj[t]
        end
        # Fill remaining with last value
        for t in (length(traj)+1):n_time
            trajectories[t, rep] = traj[end]
        end
    end

    return quasi_extinction(trajectories; threshold=threshold)
end
