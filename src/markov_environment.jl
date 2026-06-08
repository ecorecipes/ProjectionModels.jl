"""
Markov environment switching for stochastic population models.

Supports first-order Markov-structured environmental stochasticity where
the probability of choosing the next environmental state depends on the
current state (environment transition matrix).

Reference: Caswell (2001) Ch. 14; Tuljapurkar (1990).
"""

"""
    MarkovEnvironment{T<:Real, M<:AbstractMatrix}

First-order Markov environment for stochastic projections.

The environment switches between discrete states according to a transition
matrix P where P[i,j] = probability of moving to state j given currently
in state i.

# Fields
- `transition_matrix`: Row-stochastic matrix of environment transitions
- `n_environments`: Number of discrete environment states
- `stationary_distribution`: Equilibrium distribution of environments

# Example
```julia
# Two environments: good years (state 1) and bad years (state 2)
# Good years tend to persist, bad years likely followed by good
P = [0.7 0.3; 0.6 0.4]
env = MarkovEnvironment(P)
```
"""
struct MarkovEnvironment{T<:Real, M<:AbstractMatrix{T}}
    transition_matrix::M
    n_environments::Int
    stationary_distribution::Vector{T}

    function MarkovEnvironment(P::AbstractMatrix{T}) where T<:Real
        n = size(P, 1)
        size(P, 2) == n || throw(DimensionMismatch("Transition matrix must be square"))

        # Validate row-stochastic
        for i in 1:n
            row_sum = sum(P[i, :])
            abs(row_sum - 1.0) < 1e-10 || throw(ArgumentError(
                "Row $i sums to $row_sum, not 1.0. Transition matrix must be row-stochastic."))
        end

        # Compute stationary distribution (left eigenvector of P')
        F = eigen(Matrix(P'))
        idx = argmax(real.(F.values))
        π = real.(F.vectors[:, idx])
        π = abs.(π)
        π ./= sum(π)

        new{T, typeof(P)}(P, n, π)
    end
end

"""
    sample_next(env::MarkovEnvironment, current_state::Int)

Sample the next environment state given the current state.
"""
function sample_next(env::MarkovEnvironment, current_state::Int)
    probs = env.transition_matrix[current_state, :]
    r = rand()
    cumsum = 0.0
    for j in 1:env.n_environments
        cumsum += probs[j]
        if r <= cumsum
            return j
        end
    end
    return env.n_environments  # numerical safety
end

"""
    sample_initial(env::MarkovEnvironment)

Sample an initial environment state from the stationary distribution.
"""
function sample_initial(env::MarkovEnvironment)
    r = rand()
    cumsum = 0.0
    for j in 1:env.n_environments
        cumsum += env.stationary_distribution[j]
        if r <= cumsum
            return j
        end
    end
    return env.n_environments
end

"""
    simulate_environments(env::MarkovEnvironment, n_steps::Int;
                         initial_state::Union{Int,Nothing}=nothing)

Generate a sequence of environment states.
"""
function simulate_environments(env::MarkovEnvironment, n_steps::Int;
                               initial_state::Union{Int,Nothing}=nothing)
    states = zeros(Int, n_steps)
    states[1] = initial_state !== nothing ? initial_state : sample_initial(env)
    for t in 2:n_steps
        states[t] = sample_next(env, states[t-1])
    end
    return states
end

"""
    project_markov(kernels::AbstractVector{<:AbstractMatrix},
                   env::MarkovEnvironment, n0::AbstractVector, tspan::Int;
                   n_reps::Int=1)

Project population forward using Markov-switching environment.

# Arguments
- `kernels`: Vector of projection matrices, one per environment state
- `env`: MarkovEnvironment defining the switching dynamics
- `n0`: Initial population vector
- `tspan`: Number of time steps
- `n_reps`: Number of stochastic replicates

# Returns
Named tuple `(trajectories, env_sequences)` where `trajectories` is
(tspan+1 × n_reps) and `env_sequences` is (tspan × n_reps).
"""
function project_markov(kernels::AbstractVector{<:AbstractMatrix},
                        env::MarkovEnvironment, n0::AbstractVector, tspan::Int;
                        n_reps::Int=1)
    length(kernels) == env.n_environments || throw(ArgumentError(
        "Number of kernels ($(length(kernels))) must match number of environments ($(env.n_environments))"))

    n = length(n0)
    trajectories = zeros(Float64, tspan + 1, n_reps)
    env_sequences = zeros(Int, tspan, n_reps)

    for rep in 1:n_reps
        pop = Float64.(n0)
        trajectories[1, rep] = sum(pop)

        state = sample_initial(env)
        for t in 1:tspan
            env_sequences[t, rep] = state
            pop = kernels[state] * pop
            trajectories[t + 1, rep] = sum(pop)
            # Normalize to prevent overflow/underflow
            s = sum(pop)
            if s > 0
                pop ./= s
                pop .*= trajectories[t + 1, rep]
            end
            state = sample_next(env, state)
        end
    end

    return (trajectories=trajectories, env_sequences=env_sequences)
end
